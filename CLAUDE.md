# Project

**Title**: Transformer-based World Model for Crypto Trading

`[foo.bar]` denotes a key in `config.yaml`. All paths are relative to the repo root.

## Goal

Learn a latent world model over Binance USD-M **futures** candles, then train a policy on
top of its latent states that outputs a target market allocation in `[-1, 1]`
(`-1` = 100% short, `0` = flat, `+1` = 100% long), rebalanced every 15 minutes.

**Definition of done** — on the held-out `[data.test_interval]`, across all symbols, with
commission `[data.actions.commission_value]` charged on turnover:

- **Primary**: mean end-equity > 1.0 with a positive Sortino ratio, beating both
  buy-and-hold and a flat-`0` baseline. `src/test_actor.py` reports all three plus win rate.
- **Secondary**: close a meaningful fraction of the gap to the clairvoyant oracle
  (see *Oracle labels*). The oracle sees the future and is **not** attainable — the
  residual gap is expected, not a bug.

## Roadmap

| Stage | Status | What |
|-------|--------|------|
| 1. World model | Implemented | `JEPA`: encoder + AdaLN predictor + return head, SIGReg, multi-step rollout |
| 2. Imitation actor | Implemented | `Actor`: behaviour-cloning of the Sortino oracle on frozen latents |
| 2b. Autoregressive actor | Implemented, unrun | `ActorAR`: conditions on its own past allocations |
| 3. RL actor | Implemented, unrun | `ActorRL`: PPO **or** exact analytic-gradient policy search on the frozen latents, on real data or inside `JEPA.dream` |

Stage 3 is the intended end state and stage 2 is its warm start: `ActorRL` loads
`models/[actor.name]/best.ckpt` and fine-tunes it. The earlier PPO actor-critic at
`git show 27d3f9d:src/actorcritic.py` is superseded — see *Stage 3* below for what it
got wrong — but is still readable history.

## Pipeline at a glance

```
Binance API --> data/raw/{SYMBOL}.csv        (1m OHLCV, futures klines)
            --> data/processed/{SYMBOL}.csv  (causal features + Return + RetVol)
            --> CryptoDataset                (windowed tensors + cached oracle labels)
            --> JEPA        (encoder + predictor + return head)   [stage 1, frozen after]
            --> Actor       (backbone + action head)              [stage 2]
            --> ActorAR     (autoregressive fine-tune)            [stage 2b]
            --> ActorRL     (PPO / analytic-gradient fine-tune)   [stage 3]
            --> test_jepa / test_actor  (metrics.py + viz.py) -> figs/ + metrics.json
```

## Data

**Source**: `python-binance`, `HistoricalKlinesType.FUTURES`, `[data.timeframe]` = 1m,
across `[data.symbols]`. `data/` and `models/` are **gitignored** — a fresh clone has no
data. Run `uv run src/data.py` to download and preprocess.

**Processed columns**. `compute_features` is shared by the offline pipeline and any live
path, so the two cannot drift apart. Model inputs are `FEATURE_COLS` (`[jepa.input_dim]` = 6):

- `Open/High/Low/Close`: z-scored against the rolling mean/std **of Close** over
  `[data.normalization_window]` = 1440 bars.
- `Volume`: `log1p`, then rolling z-score over the same window.
- `Volatility`: `log(rv_short / rv_long)` — realised vol of log returns over
  `[data.vol_window_short]` vs the full window. A *ratio*, so it is O(1) and comparable
  across symbols. (It used to be `std/mu` of price, which measures trend dispersion, not
  return volatility.)

Not model inputs:

- `Return`: raw 1m `pct_change`. Everything economic uses this.
- `RetVol`: realised vol of 1m log returns, **shifted one bar** so the normaliser for a
  window is known strictly before that window begins. Verified causal by a spike test.

### Windowing — read this before reasoning about horizons

1. `[data.window_size]` = 15 **non-overlapping** 1m bars aggregate into one *window*.
   **The world model's timestep is 15 minutes, not 1 minute.**
2. `[data.sequence_length]` = 64 consecutive windows form one sequence (= **16 hours**),
   at `[data.stride]` = 32 for training (50% overlap, cheap augmentation) and
   `[data.eval_stride]` = 64 for val/test (**disjoint**, so metrics do not average
   correlated duplicates).

Sample shape is `[64, 15, 6]`.

### Two return series, deliberately

- `return_raw` — compounded 15m return, real units. Equity, commission and the oracle
  labels all use this.
- `return` — `return_raw / (RetVol_at_window_start * sqrt(15))`, i.e. **in units of
  sigma**. This is what the model conditions on and predicts.

The reason: prices are z-scored precisely so symbols can share a model, but a raw percent
return is not comparable across them. Measured 15m sigma is 0.347% (BTC), 0.443% (ETH),
0.580% (XRP), 0.657% (SOL) — SOL moves 1.9x BTC. Feeding raw percent made the return head
fit a mixture of differently-scaled distributions with no symbol identifier to
disambiguate. After normalisation the pooled series has std 1.045 and only 0.46% of
windows clip at the +-5 sigma bin range.

`batch['vol']` carries the normaliser so sigma-space predictions can be converted back to
real returns (`test_jepa.py` does this to plot cumulative return).

### Discretisation

Both models are **classifiers**. `util.preprocess_classes` clips *and* bins; `bin_centers`
is the matching decode table.

- Returns: `[data.returns.num_bins]` = 61 bins over +-5 sigma.
- Actions: `[data.actions.num_bins]` = 51 bins over [-1, 1].

### Oracle labels (the imitation target)

`util.optimal_allocation` optimises an allocation path `x = tanh(u)` with Adam
(`[data.actions.opt_steps]`) against `make_constrained_loss(loss_fn_so)`:

- `loss_fn_so`: negative **Sortino** ratio — mean excess step return over the downside
  deviation, where downside deviation is RMS shortfall below break-even.
- constraint: quadratic penalty on per-step allocation changes above
  `[data.actions.max_change]` — a turnover regulariser.
- `delta_equity(x, p, c) = x[1:] * p - c * |diff(x)| + 1`, `c = 0.0005` (5 bps on turnover).

Alignment: `actions[t]` applies to `returns[t]`, so the allocation is chosen *before* that
return is realised. The oracle sees the whole 64-step future — a clairvoyant upper bound
used only as a behaviour-cloning target.

Labels cost ~1000 Adam steps per symbol, so they are **cached** under `data/cache/`, keyed
by symbol, split, window/stride geometry, commission, and a fingerprint of the CSV slice.
Delete that directory to force recomputation.

### Splits

`[data.training_interval]` / `validation_interval` / `test_interval` are disjoint date
ranges (train <= 2025 < val <= 2026 < test <= 2026-06). No purging/embargo at the
boundary; sequences are 16h, so the leak is bounded and currently ignored.

## Models

### Stage 1 — `JEPA` (`src/jepa.py`)

- **Encoder**: CLS token + `[jepa.encoder.num_layers]` pre-LN layers over the 15 candles of
  one window; the CLS row (after a final LayerNorm) is the latent `Z_t`.
- **Predictor**: causal transformer over latents, **AdaLN**-conditioned on the current
  window's sigma-return. Predicts `Z_{t+1}` from `Z_{<=t}`.
- **Return head**: MLP -> 61-way logits for the next window's binned sigma-return.

Loss = `lam_state * MSE + lam_CE * CE + lam_SIGReg * SIGReg + lam_rollout * rollout`.

- **SIGReg** (`modules.SIGReg`) is the anti-collapse term — a sketched Epps-Pulley
  Gaussianity statistic over random projections of the latents (LeJEPA). It matters
  because `Z_target` comes from the *same online encoder* with no stop-gradient and no EMA
  target, so the MSE term alone has the trivial solution "encoder emits a constant". Set
  `[jepa.lam_SIGReg]` to 0 and you reproduce the collapsing setup. `val/latent_std` and
  `val/latent_erank` are logged so collapse is visible rather than inferred.
- **Rollout** (`JEPA.rollout`) runs `[jepa.training.rollout_steps]` ahead feeding predicted
  latents and the head's *expected* return back in, and supervises every step. Teacher
  forcing alone leaves the predictor unprepared for `dream()`, which is exactly where the
  RL stage will live. Gradients flow through the whole rollout; targets are detached so
  this term trains the predictor rather than adding a second route to collapse. Costed by
  `[jepa.training.rollout_batch]`, since k sequential passes are all held for backprop.
  `[jepa.lam_rollout]` = 0 recovers pure teacher forcing.

`JEPA.dream(...)` is the inference-time open-loop rollout, sampling returns from the head.
Its outputs are in **sigma units** — multiply by `vol` for real returns.

### Stage 2 — `Actor` (`src/actor.py`)

The JEPA is loaded from `./models/[jepa.name]/best.ckpt`, `eval()`-ed and **frozen**.

Per step the actor receives the predicted next latent `Z_hat_{t+1}` as input, and as AdaLN
condition the concatenation of:

- the world model's predicted return distribution (61 bins), and
- **the previous allocation** (1).

The previous allocation is load-bearing: commission is charged on `|dx|`, so a policy that
cannot see its current position is making a cost-aware decision blind. It also makes
`ActorAR` a drop-in rather than a separate model.

`decode_actions` turns bin probabilities into an allocation. Default is `'expected'` (the
probability-weighted mean) — allocation is continuous, so this is both a sensible decoder
and a deterministic one, which matters because `val/mean_eq` is the checkpoint *and*
early-stopping monitor. `'sample'` and `'argmax'` are available.

**`ActorAR`** fine-tunes the same network on its own past allocations instead of the
oracle's, closing the behaviour-cloning train/test gap. It rolls forward on *real* market
latents and returns — only the action history is self-generated.

#### Training is teacher-forced; evaluation is not

Training conditions on the oracle's previous allocation (ordinary behaviour cloning).
**`backtest` must not.** `batch['action']` is the clairvoyant oracle path, fitted with Adam
over the whole 64-step future, so conditioning on it at eval time (a) leaks the future —
the backbone is causal, so position `t` sees oracle allocations `0..t` — and (b) charges
commission against a position path the policy never saw, since `_equity_bundle` diffs the
*actor's own* actions. The oracle path is smooth by construction (`max_change`), so its lag
is close to a copy of the label: a network that learns the identity map off the condition
scores well having learned nothing.

So `Actor.rollout` (shared with `ActorAR`, which additionally *trains* through it) feeds
back the policy's own allocations, starting flat, and `Actor.backtest` uses it by default —
`[actor.eval.ctx_len]` = 1 real windows of context, then self-feeding to the end
(`[actor.eval.pred_steps]` = `null`). `[actor.eval.val_pred_steps]` caps the in-training
monitor, since each step is a sequential forward pass and validation runs every epoch.
Both actors are scored over the same span, so the two are actually comparable.

`backtest(teacher_force=True)` reproduces the leaky oracle-conditioned pass. It is logged
as `val/mean_eq_tf` and available via `test_actor.py --teacher-force` **as a diagnostic
only** — the gap to the honest number *is* the behaviour-cloning train/test gap, which is
the thing `ActorAR` exists to close. Never report it as a result.

### Stage 3 — `ActorRL` (`src/rl.py`)

Subclasses `Actor`, so the network, the decoder, `_equity_bundle` and `backtest` are
the same objects; only the training signal changes. Warm-started from the stage-2
checkpoint (`[rl.from_ar]` switches to the stage-2b one).

**Three structural facts drive every default here**, and they are why this is not a
generic RL setup:

1. **The reward is closed-form and differentiable in the action.**
   `delta_equity(x, p, c) = x_t*p_t - c*|dx| + 1`. Nothing about it has to be learned.
2. **Actions do not affect the dynamics.** An allocation does not move the market, so
   `(Z_t, r_t) -> (Z_{t+1}, r_{t+1})` is exogenous and the only action-dependent state
   is the position, which is observed and deterministic. Therefore:
   - On-policy data is **free on real market data**. The usual reason to roll out
     inside a world model — expensive environment interaction — does not apply, so
     `[rl.env]` defaults to `'real'`. `'dream'` and `'mixed'` exist, but if the return
     head is only marginally better than the unconditional distribution then a dreamed
     market is a near-iid draw from the marginal, and the best policy against that is
     a constant. Dreams also need a `vol` normaliser the model does not predict, so it
     is held at its last real value — a real, documented bias.
   - **Credit assignment is two steps deep.** `a_t` enters `r_t` and, through
     `c|a_{t+1} - a_t|`, `r_{t+1}`; beyond that it acts only via the policy's own later
     choices. GAE at `lambda ~ 0.95` over 64 steps therefore pours in market noise that
     `a_t` had no influence over, which is why `[rl.gae_lambda]` defaults to **0.0**,
     not the usual value. `[rl.gamma]` is **1.0** and is not a tuning knob: the
     undiscounted sum of per-step log growth *is* log terminal equity.
     `[rl.ppo.center_adv]` subtracts the batch mean at each timestep, a free
     time-varying baseline the zero-initialised critic cannot supply.
3. **Collection is sequential; the update is not.** The policy's only autoregressive
   input is its previous allocation, and during the update that path is stored data —
   so re-scoring a whole episode is ONE teacher-forced pass, not T recurrent ones
   (`_replay_inputs` / `_score`). The removed `actorcritic.py` flattened transitions
   into length-1 sequences for its update, discarding the transformer's context
   entirely. `smoke_test.py` asserts the replay reproduces the collection logits
   exactly; if that ever stops holding, every PPO ratio is wrong.

**Two estimators**, `[rl.objective]`:

- `ppo` — clipped surrogate on sampled bins, GAE advantage, value head on the shared
  backbone (zero-initialised, since true `V` is near 0 here; `[rl.critic.warmup_updates]`
  trains it alone before it is allowed to move the policy).
- `analytic` — because (1) and (2) hold, the gradient of terminal log-equity w.r.t.
  the policy is **exact**: backprop straight through the differentiable `expected`
  decode. No critic, no policy-gradient variance. This is `util.optimal_allocation`
  with the allocation parameterised by a causal network instead of free clairvoyant
  variables. `[rl.analytic.objective]` picks log-equity or Sortino. It is a direct
  policy search, so expect it to overfit the training split harder than PPO.

#### The reward is shaped, and that is what stops the buy-and-hold collapse

The first RL run collapsed to always-long. That was not a bug — it was the correct
answer to the question being asked. `sum_t log dE_t` over the training split is
maximised by a *constant* allocation of about **+0.85** (measured by sweeping constant
`a` on real BTC 15m windows; the Kelly fraction `mu/sigma^2` is 1.08 before costs, which
the `[-1, 1]` box clips). Measured drift per 15m window is **+0.325 bp on train, +0.045
bp on val, −0.171 bp on test** — so the sign *flips* out of sample, and a policy that
learned the drift is anti-fitted to the test split. Three settings in `[rl.reward]`
address this, and they are not interchangeable:

- **`scale`** = 10000: rewards are log growth in **basis points**. Raw log growth is
  O(1e-4), so at scale 1.0 every coefficient below, plus `kl_coef` and `ent_coef`, sat
  orders of magnitude above the objective it was meant to regularise. (`analytic.kl_coef`
  of 0.02 against an objective of O(1e-3) was ~20x the whole objective — that run could
  not move.) The two shaping terms are now **linear** in those units: `downside_coef`
  = 1.0 makes a bp lost count twice a bp gained; `turnover_penalty` is extra bps per unit
  of change beyond `max_change`, the same units as `[data.actions.commission_value]`.
  A quadratic cannot be scale-coherent here — squaring a bps quantity leaves the
  coefficient carrying the units.
- **`demean`** = `'ema'`: subtract a running mean of the realised window return before
  computing the reward, so a *constant* allocation earns ~0 and only timing pays. This is
  the setting that moves the argmax; with it the best constant `a` is **0.00 on both
  train and test**. The estimate is built from the training stream and applied only to
  the training reward — `backtest`, and every reported number, still runs on real returns
  in real units, so this is shaping, not look-ahead.
- **`benchmark`** = `'bh'`: score `log(dE / dE_bh)`, so an episode return is
  `log(E_actor / E_bh)` — the primary definition of done — and buy-and-hold scores
  *exactly* 0 rather than winning. This is the baseline PPO's critic cannot supply, since
  true `V` really is ~0 here. It does **not** move the analytic-logeq optimum: dividing by
  a path the actions do not enter is a constant offset in log space. Use both.

`smoke_test.py` pins all three: the unshaped reward still sums to log terminal equity,
`dE_raw` still reproduces the backtest path exactly, always-long scores exactly zero
under the benchmark, and demeaning neutralises a constant allocation on a trending
stream.

**The other load-bearing regulariser is `[rl.*.kl_coef]`, not the entropy bonus.** Entropy
over 51 *ordered* bins pulls toward uniform on [-1, 1], a nonsense prior for an
allocation. The KL is measured against a frozen copy of the behaviour-cloned policy —
a risk-aware prior distilled from the Sortino oracle. The reference lives in the state
dict (so `--resume` keeps it) and is pinned by `sync_reference()` at launch, guarded by
the `ref_synced` buffer.

`[rl.disable_dropout]` is on by default: PPO's ratio is a trust region only if
collection and update see the same network, and dropout makes `new_logp != old_logp`
at epoch 0 for identical weights.

Evaluation is `Actor.backtest`, inherited **unchanged**, so the RL policy is scored on
the same honest autoregressive rollout over the same `[actor.eval]` span as `Actor` and
`ActorAR`. **`val/excess_eq` — mean `log(E_actor / E_bh)` — is the checkpoint and
early-stopping monitor** for both `ActorAR` and `ActorRL`. `val/mean_eq` is still logged
but must not be selected on: a policy that outputs +1 everywhere *is* buy-and-hold and
scores exactly `val/bh_eq`, so selecting on raw equity rewards the collapse it is
supposed to catch. `val/mean_alloc` is the collapse detector — read it first. `val/bc_ce`
logs cross-entropy against the oracle as a *drift* diagnostic — it is expected to get
worse if RL is finding anything the clairvoyant labels do not contain.

`ActorRL` uses manual optimisation (PPO takes several optimiser steps per collected
batch), so `build_trainer` must be given `gradient_clip_val=None` for it — Lightning
raises rather than ignores automatic clipping in that mode — and `ActorRL` clips
inside its own update instead.

## Running things

`uv`-managed (Python >= 3.14, torch on the `cu126` index).

```
uv run src/smoke_test.py             # shape/wiring check, seconds, CPU -- run this first
uv run src/data.py                   # download + preprocess
uv run src/main.py jepa    [--resume]
uv run src/main.py actor   [--resume]
uv run src/main.py both    [--resume]
uv run src/main.py actor-ar[--resume]
uv run src/main.py rl      [--resume]  # stage 3, warm-started from the actor
uv run src/test_jepa.py              # world-model eval -> figs/dreams/
uv run src/test_actor.py [--ar|--rl] # trading eval     -> figs/backtest/
```

`test_actor.py` also takes `--ckpt {best,last}`, `--decode {expected,argmax,sample}` and
`--teacher-force` (diagnostic; writes to `figs/backtest-teacher-forced/` so a leaky
`metrics.json` can never be mistaken for the real one).

## Evaluation

`src/metrics.py` holds the scoring functions (pure numpy, unit-tested in
`smoke_test.py`); `src/viz.py` holds the shared plot style.

**`test_jepa.py`** — is the world model worth anything? Writes `metrics.json` and:

| Figure | Question it answers |
|---|---|
| `horizon_skill.png` | Does the rollout beat persistence, and for how many steps? |
| `calibration.png` | Reliability diagram + PIT histogram — are the probabilities honest? |
| `confusion.png` | Predicted vs realised return bin |
| `return_dist.png` | Do dreamed returns have the right distribution? |
| `dream{i}.png` | Sample open-loop rollouts against the true path |

Headline numbers are **skill scores, not raw losses**. A latent MSE depends entirely on
the scale the encoder settles at, and an NLL on the bin count — neither is readable alone.
So: `bits_gained` over the empirical marginal (<= 0 means nothing was learned about the
conditional), latent MSE over a persistence baseline, `directional_acc` against 0.5, and
`latent_erank` against `d_model`. Also RPS/CRPS, which unlike accuracy reward being *close*
— relevant when there are 61 ordered bins and top-1 is a harsh score.

**`test_actor.py`** — does the policy make money? Writes `metrics.json` and:

| Figure | Question it answers |
|---|---|
| `equity_summary.png` | Median equity + IQR band, actor vs oracle vs buy-and-hold |
| `outcomes.png` | End-equity and max-drawdown distributions |
| `allocations.png` | Allocation distributions, and actor-vs-oracle agreement |
| `per_symbol.png` | ROI by symbol vs buy-and-hold |
| `backtest{i}.png` | Individual sequences |

Reports ROI, annualised Sharpe/Sortino, max drawdown, Calmar, win rate, profit factor,
turnover, commission drag and long/short/flat exposure — each for the actor, buy-and-hold
and the oracle — plus a **bootstrap CI and t-stat on mean ROI**, and a **per-symbol
breakdown**. Both matter: a positive mean over a few hundred correlated 16h windows is not
on its own evidence of anything, and a pooled number can hide a policy that only works on
one market. The bootstrap resamples *sequences*, not steps, so within-sequence
autocorrelation is respected.

Annualised figures pool step returns across disjoint 16h windows — indicative, not a
continuous track record.

**Import constraint**: `src/` is not a package — modules import each other flatly
(`from jepa import JEPA`). Always launch as `uv run src/main.py`; `python -m src.main` fails.

**Config constraint**: `config.yaml` uses a custom OmegaConf `eval` resolver. Every entry
point must call `OmegaConf.register_new_resolver("eval", eval, replace=True)` *before*
`OmegaConf.to_container(cfg, resolve=True)`.

**Checkpoint constraint**: models call `save_hyperparameters()` (and `Actor` ignores
`jepa`), so hparams hold a single `cfg` key. Passing the config dict positionally would
splat its top-level keys — one of which is `jepa` — and collide with `Actor`'s `jepa`
argument on load.

**Logging**: Weights & Biases, entity `rudyhuy`, project `[experiment_name]`. The run id is
persisted to `models/<name>/wandb_id.txt` so `--resume` rejoins the right run.

**Cluster**: `jobs/*.sh` are LSF scripts for the DTU HPC (`bsub < jobs/jepa.sh`). They
hardcode `/zhome/d3/0/155487/worldmodel` and a v100 32GB queue.

**Local dev constraint**: this used to say that Windows Application Control blocked
torch's DLLs (`c10.dll`) on the Windows dev machine. As of 2026-09-06 that is no longer
true — `uv run src/smoke_test.py` runs there on CPU, torch 2.13.0+cu126, in about two
minutes. Training still belongs on the cluster (no local GPU), but wiring changes can
and should be smoke-tested locally first.

## Known issues / open work

1. Nothing has been retrained since the architecture and data changes, so **no result in
   `figs/` or any checkpoint under `models/` is current**. `[jepa.d_model]` changed from 64
   to 256 — old checkpoints will not load.
2. `ActorAR` has never completed a run. It is shape-checked by `smoke_test.py` only.
3. No purging/embargo at split boundaries.
4. No unit tests beyond `smoke_test.py` (which covers shapes end-to-end and the
   `metrics.py` scoring functions numerically); no `README.md`.
5. `ActorRL` has never been run on real data. `smoke_test.py` covers it end to end on
   synthetic data (both estimators, through a real Lightning `fit`), but nothing has
   been trained.
6. **`util.loss_fn_so` is not the fixed version this file describes below.** The live
   body still thresholds on `dE < 0`; the corrected shortfall-below-break-even version
   sits next to it inside a string literal. Every cached oracle label under
   `data/cache/` was therefore produced by the broken objective. `rl.sortino_ratio` is
   the correct implementation. Fixing `loss_fn_so` invalidates the label cache and
   means retraining stages 2 / 2b / 3.

## Hyperparameter review

Changed, with reasons — all still unswept, these are informed starting points, not tuned
values:

| Parameter | Was | Now | Why |
|---|---|---|---|
| `jepa.d_model` | 64 | 256 | 72 layers at width 64 is ~2.4M params behind a 72-layer critical path; `predictor.num_heads` 16 gave `head_dim` **4**, degenerate for attention |
| `encoder.num_layers` | 24 | 6 | encoding 16 tokens does not need 24 layers |
| `predictor.num_layers` | 48 | 8 | as above; `head_dim` is now 32 |
| FFN expansion | 2x | 4x | 2x is unusually narrow |
| `returns.min/max` | +-0.03 (percent) | +-5.0 (sigma) | +-3% was +-4.6 to +-8.6 sigma depending on symbol; ~77% of bins were near-empty |
| `epochs` / `patience` | 2000 / 100 | 60 / 10 | the cosine schedule spans `max_epochs`; stopping at ~5% of it meant the LR never left its warmup plateau |
| `warmup` | epochs-derived | explicit steps | the old expression was warmup-in-epoch-units scaled by total steps |
| head dropout (2nd layer) | `2*dropout` = 0.2 | `dropout` | looked like a copy-paste of the neighbouring `2*d_model` |
| `eval_stride` | (none) | 64 | val/test windows are now disjoint |
| `rl.reward.scale` | (none) | 1e4 | rewards in bps; at 1.0 every shaping/KL coefficient was orders of magnitude off the objective |
| `rl.reward.demean` | (none) | `'ema'` | constant-long was the exact argmax of the training objective — see *The reward is shaped* |
| `rl.reward.benchmark` | (none) | `'bh'` | the episode return is now `log(E/E_bh)`, i.e. the definition of done, and PPO's missing baseline |
| `rl.gae_lambda` | 0.5 | 0.0 | credit is two steps deep; anything longer is market noise the action had no say in |
| `rl.ppo.center_adv` | (none) | true | per-timestep batch baseline; the zero-init critic supplies none |
| `rl.analytic.kl_coef` | 0.02 | 1.0 | was ~20x the entire objective at the old reward scale |
| RL monitor | `val/mean_eq` | `val/excess_eq` | raw equity cannot separate skill from drift; a long-only policy maximises it by construction |

Still worth sweeping: `d_model` and depth (the above is a guess, not a measurement);
`lam_rollout` and `rollout_steps`; `lam_SIGReg`; the +-5 sigma bin range (16 of 61 bins
carry >1% of mass, so non-uniform or quantile bins are an option — note `util.symlog`
already exists and is unused); `max_change` and the oracle's turnover penalty.

For stage 3, nothing is tuned at all. The knobs most likely to matter, in order:
`[rl.gae_lambda]` (the argument above says short; 0.0—0.3 is the range worth
measuring), `[rl.ppo.kl_coef]` (too low and the policy degenerates, too high and it
never leaves the clone), `[rl.training.lr]`, `[rl.objective]` itself, and
`[rl.reward.downside_coef]` if the Sortino side of the definition of done lags.

## Bugs fixed in the last pass (do not reintroduce)

- **`Actor.backtest` conditioned on the oracle's previous allocation.** Future leakage, not
  merely exposure bias — and `val/mean_eq`, the checkpoint *and* early-stopping monitor, ran
  on it, so model selection itself was contaminated. Backtests are autoregressive now; see
  *Training is teacher-forced; evaluation is not*. Every number in `figs/backtest/` from
  before this change is void.
- **`Actor` and `ActorAR` were scored over different spans** (`1..S-1` vs `ctx_len..ctx_len+
  pred_steps`), so the comparison between them measured horizon as much as policy. Both now
  inherit one `backtest` driven by `[actor.eval]`.
- `ActorAR.rollout` capped `pred_steps` at `S - 1 - ctx_len`, one step short of the
  sequence; it is `S - ctx_len`.
- **`loss_fn_so` was not Sortino.** It thresholded on `dE < 0`, but `dE` is a gross return
  factor near 1.0, so the mask was almost always empty, `downside_std` collapsed to 0 and
  the ratio blew up. It was effectively maximising mean return with no risk term at all.
  Now measured as RMS shortfall below break-even. `loss_fn_sh` had the same missing
  reference point.
- **Bin decode was off by half a bin.** `preprocess_classes` used `num_bins+1` edges while
  decoding used `linspace(min, max, num_bins)` — both the spacing and the offset were
  wrong, by up to ~0.15% return. Use `bin_centers`.
- `modules.Predictor.forward` truncation (`x[:, -max_len, :]`, missing colon) silently
  dropped a dimension; `cond` was not truncated alongside `x`.
- `ActorAR` built a 62-dim condition against a 61-dim `condition_dim`.
- Oracle labels were recomputed on every run, including JEPA runs that discard them.
- A single wandb run id was hardcoded into both resume paths.
- Actor checkpointed on `val/mean_eq` but early-stopped on `val/actor_loss`.
- `pyproject.toml` depended on `logging` — a Python-2-era PyPI package that shadows the
  stdlib module. It only worked because site-packages sorts after stdlib on `sys.path`.

## Working preferences

- Help me **finish** this project — take it to a validated result on the test split.
- Use the current implementation as the starting point, but don't be afraid to change it.
  **Give me a detailed heads-up before changing anything, saying what and why.**
- Some comments and log strings are in Danish. Keep new code comments in English.
- Match the existing style: flat imports, Lightning modules, config-driven, `logger.info`
  over `print`.
