"""Stage 3 -- reinforcement learning on the frozen world model's latents.

This MDP is not a generic one, and most of the defaults below follow from three
structural facts rather than from convention.

* **The reward is known in closed form and differentiable in the action.**
  `delta_equity(x, p, c) = x_t * p_t - c * |dx| + 1`. There is no reward model
  to learn and no environment to query.

* **Actions do not affect the dynamics.** An allocation does not move the
  market, so `(Z_t, r_t) -> (Z_{t+1}, r_{t+1})` is exogenous and the only
  action-dependent state is the position itself, which is observed and
  deterministic. Two consequences run through this file:

    - On-policy data is free on *real* market data, so `[rl.env]` defaults to
      `'real'`. The usual reason to roll out inside a world model -- that
      environment interaction is expensive -- does not apply. `'dream'` is
      implemented because it is on the roadmap, but if the return head is only
      marginally better than the unconditional distribution then dreamed
      markets are near-iid draws from the marginal, and the best policy against
      that is a constant.

    - Credit assignment is *two steps* deep. `a_t` enters `r_t` (through
      `a_t * p_t` and `c|a_t - a_{t-1}|`) and `r_{t+1}` (through
      `c|a_{t+1} - a_t|`); past that it acts only through the policy's own
      later choices. GAE over 64 steps at lambda ~ 1 therefore pours in market
      noise that `a_t` had no influence over, which is why `[rl.gae_lambda]`
      defaults low rather than to the usual 0.95.

* **Collection is sequential, the update is not.** The policy's only
  autoregressive input is its previous allocation, and during the update that
  path is stored data -- so re-scoring a whole episode is ONE teacher-forced
  pass over the sequence, not T recurrent ones. (The removed `actorcritic.py`
  flattened transitions into length-1 sequences for its update, which threw the
  transformer's context away entirely.)

Two estimators share all of that machinery, selected by `[rl.objective]`:

  `ppo`       clipped surrogate on sampled bins with a GAE advantage and a
              value head on the shared backbone.
  `analytic`  because the reward is differentiable and the transition is
              action-independent, the gradient of terminal log-equity w.r.t.
              the policy is *exact*. No critic, no policy-gradient variance.
              This is `util.optimal_allocation` with the allocation
              parameterised by a causal network instead of free clairvoyant
              variables.

The regulariser that matters most is not the entropy bonus. Entropy over 51
ordered bins pulls the policy toward uniform on [-1, 1], which is a nonsense
prior for an allocation. The KL penalty back to the frozen behaviour-cloned
policy is the real control: that reference is a risk-aware policy distilled
from the Sortino oracle, and it is what stops a run collapsing onto the
attractor RL loves here -- all-in long, because the training split trends up.

Evaluation is `Actor.backtest`, inherited unchanged, so the RL policy is scored
on exactly the same honest autoregressive rollout as `Actor` and `ActorAR`,
over the same `[actor.eval]` span. `val/mean_eq` remains the checkpoint and
early-stopping monitor.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from actor import Actor
from modules import MultiHeadSelfAttention
from util import delta_equity, truncate


def sortino_ratio(dE, eps=1e-8):
    """Per-sequence Sortino on a path of gross growth factors.

    Downside deviation is the RMS shortfall below break-even (`dE == 1`), not
    the dispersion of the negative values: `dE` is a growth factor near 1.0, so
    thresholding on `dE < 0` leaves an almost always empty mask and the ratio
    blows up. See the note in CLAUDE.md.
    """
    excess = dE - 1.0
    shortfall = torch.clamp(-excess, min=0.0)
    downside = torch.sqrt(torch.mean(shortfall ** 2, dim=-1) + eps)
    return excess.mean(dim=-1) / (downside + eps)


class ActorRL(Actor):
    """PPO / analytic-gradient fine-tune of the behaviour-cloned policy.

    Subclasses `Actor` so the network, the decoder, the equity bundle and the
    backtest are literally the same objects. Only the training signal changes.
    """

    def __init__(self, cfg, jepa):
        super().__init__(cfg, jepa)
        rcfg = cfg['rl']

        self.objective = rcfg['objective']
        assert self.objective in ('ppo', 'analytic'), self.objective
        self.env_mode = rcfg['env']
        assert self.env_mode in ('real', 'dream', 'mixed'), self.env_mode
        self.dream_prob = rcfg.get('dream_prob', 0.5)
        self.dream_temp = rcfg.get('dream_temp', 1.0)

        self.rl_ctx_len = rcfg['ctx_len']
        self.rl_pred_steps = rcfg['pred_steps']
        self.act_temp = rcfg.get('act_temp', 1.0)

        self.gamma = rcfg['gamma']
        self.gae_lambda = rcfg['gae_lambda']

        rw = rcfg['reward']
        # Rewards are log growth expressed in BASIS POINTS. Log growth over a
        # 15m window is O(1e-4), so at scale 1.0 every shaping coefficient here
        # -- and the KL and entropy terms further down -- sits orders of
        # magnitude above the objective it is supposed to regularise. In bps the
        # per-step reward is O(0.3) in the mean and O(40) in the spread, and a
        # coefficient of 1.0 means roughly what it looks like it means.
        self.reward_scale = rw.get('scale', 1e4)

        # (A) Unconditional drift removal. Without it the training split's drift
        # IS the objective: the measured Kelly fraction mu/sigma^2 over
        # [data.training_interval] is 1.08, which the [-1, 1] box clips to +1,
        # so "always fully long" is the exact argmax of undiscounted log equity.
        # It is -1.60 over [data.test_interval], where the drift changes sign,
        # so a policy that has learned the drift is anti-fitted to the test set.
        # Subtracting a running mean of the realised window return makes a
        # CONSTANT allocation earn ~0 and leaves only timing to be paid for.
        self.demean = rw.get('demean', 'ema')
        assert self.demean in ('none', 'ema', 'const'), self.demean
        self.drift_beta = rw.get('drift_beta', 0.99)
        self.drift_const = rw.get('drift', 0.0) or 0.0

        # (B) Benchmark. 'bh' scores log(dE / dE_buy_and_hold), so an episode
        # return is log(E_actor / E_bh) -- the primary definition of done -- and
        # buy-and-hold scores exactly 0 instead of winning. For PPO this is the
        # baseline the critic cannot supply, since true V really is ~0 here.
        # It does NOT move the analytic-logeq optimum: dividing by a path the
        # actions do not enter is a constant offset in log space. Drift removal
        # is what moves that one. The two are complementary, not alternatives.
        self.benchmark = rw.get('benchmark', 'bh')
        assert self.benchmark in ('none', 'bh'), self.benchmark

        self.downside_coef = rw.get('downside_coef', 0.0)
        self.reward_max_change = rw.get('max_change', None)
        self.turnover_penalty = rw.get('turnover_penalty', 0.0)

        p = rcfg['ppo']
        self.ppo_clip = p['clip']
        self.ppo_epochs = p['epochs']
        self.minibatch = p['minibatch']
        self.vf_coef = p['vf_coef']
        self.norm_adv = p.get('norm_adv', True)
        self.center_adv = p.get('center_adv', True)
        self.target_kl = p.get('target_kl', None)

        a = rcfg['analytic']
        self.analytic_objective = a.get('objective', 'logeq')
        assert self.analytic_objective in ('logeq', 'sortino'), self.analytic_objective
        # 'logeq' is per-step basis points, O(0.3); 'sortino' is a dimensionless
        # per-step ratio, O(0.01) on this data. They are two orders of magnitude
        # apart, so switching between them means re-scaling the KL and entropy
        # coefficients with them -- obj_scale is the knob for doing that in one
        # place rather than in three.
        self.obj_scale = a.get('obj_scale', 1.0)

        # Entropy / reference-KL coefficients are per-objective: PPO needs a
        # live entropy term to keep sampling alive, while the analytic path is
        # deterministic and mostly needs the prior.
        src = p if self.objective == 'ppo' else a
        self.ent_coef = src.get('ent_coef', 0.0)
        self.kl_coef = src.get('kl_coef', 0.0)

        c = rcfg['critic']
        self.detach_trunk = c.get('detach_trunk', False)
        self.critic_warmup = c.get('warmup_updates', 0)

        d_model = cfg['actor']['d_model']
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.LayerNorm(2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, 1),
        )
        # True V is near zero here -- the market is close to unpredictable, so
        # the sum of future log returns carries little signal. Starting the
        # critic at exactly 0 keeps it from injecting noise into the advantage
        # before it has learned anything.
        nn.init.zeros_(self.critic_head[-1].weight)
        nn.init.zeros_(self.critic_head[-1].bias)

        # Frozen behaviour-cloned reference for the KL prior. Built here (not
        # lazily) so it lives in the state_dict and survives --resume; the
        # weights are copied in by sync_reference() once the BC checkpoint has
        # actually been loaded, guarded by the buffer below.
        if self.kl_coef > 0:
            self.ref_backbone = copy.deepcopy(self.backbone)
            self.ref_head = copy.deepcopy(self.actor_head)
            for q in list(self.ref_backbone.parameters()) + list(self.ref_head.parameters()):
                q.requires_grad = False
        else:
            self.ref_backbone = self.ref_head = None
        self.register_buffer('ref_synced', torch.zeros((), dtype=torch.long))

        if rcfg.get('disable_dropout', True):
            self._disable_dropout()

        # PPO runs several optimiser steps per collected batch, so Lightning
        # cannot drive them. build_trainer must pass gradient_clip_val=None in
        # this mode -- automatic clipping is incompatible with manual
        # optimisation and Lightning raises rather than ignoring it.
        self.automatic_optimization = False

        tcfg = rcfg['training']
        self.lr = tcfg['lr']
        self.weight_decay = tcfg['weight_decay']
        self.warmup_steps = tcfg['warmup_steps']
        self.sched_steps = tcfg.get('sched_steps', None)

        self.register_buffer('updates', torch.zeros((), dtype=torch.long))
        # Running estimates: the unconditional drift (A) and the spread of the
        # GAE target (D). Buffers, so --resume continues with a warmed-up
        # estimate instead of re-learning it from a cold start. The paired
        # `_init` flags let the first batch seed them directly -- an EMA at
        # beta = 0.99 started from zero takes hundreds of steps to catch a
        # quantity as small as 3e-5.
        self.register_buffer('drift_ema', torch.zeros(()))
        self.register_buffer('drift_init', torch.zeros((), dtype=torch.long))
        self.register_buffer('ret_std', torch.ones(()))
        self.register_buffer('ret_init', torch.zeros((), dtype=torch.long))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _disable_dropout(self):
        """Switch dropout off for the whole RL stage.

        PPO's ratio is a trust region only if collection and update see the same
        network. With dropout on, `new_logp != old_logp` at epoch 0 even for
        identical weights, so the ratio starts as noise and the clip stops
        meaning what it says. The reference KL is the regulariser here instead.
        """
        for m in (self.backbone, self.actor_head, self.critic_head):
            for sub in m.modules():
                if isinstance(sub, nn.Dropout):
                    sub.p = 0.0
                elif isinstance(sub, MultiHeadSelfAttention):
                    sub.dropout_p = 0.0

    def sync_reference(self):
        """Copy the current policy into the frozen KL reference."""
        if self.ref_backbone is None:
            return
        self.ref_backbone.load_state_dict(self.backbone.state_dict())
        self.ref_head.load_state_dict(self.actor_head.state_dict())
        for q in list(self.ref_backbone.parameters()) + list(self.ref_head.parameters()):
            q.requires_grad = False
        self.ref_synced.fill_(1)

    def on_fit_start(self):
        # Fallback for a run that forgot to sync explicitly. On --resume the
        # checkpoint is restored before this hook, so ref_synced is already 1
        # and the saved reference is left alone.
        if self.ref_backbone is not None:
            if int(self.ref_synced) == 0:
                self.sync_reference()
            self.ref_backbone.eval()
            self.ref_head.eval()

    # ------------------------------------------------------------------
    # Rollout / environment
    # ------------------------------------------------------------------

    def _use_dream(self):
        if self.env_mode == 'real':
            return False
        if self.env_mode == 'dream':
            return True
        return bool(torch.rand(()) < self.dream_prob)

    def _rollout(self, batch, decode, detach_actions, dream):
        if dream:
            return self._dream_rollout(batch, decode, detach_actions)
        return self.rollout(batch, self.rl_ctx_len, self.rl_pred_steps,
                            act_temp=self.act_temp, decode=decode,
                            detach_actions=detach_actions)

    def _dream_rollout(self, batch, decode, detach_actions):
        """Roll the policy forward inside the world model.

        Identical bookkeeping to `Actor.rollout`, but the latents and returns
        after the prompt are the model's own. The return head speaks in sigma
        units and the vol normaliser is not part of the world model, so it is
        held at its last real value for the whole horizon -- a bias, bounded
        over the horizon the default covers, but a real one.
        """
        X, Ret = batch['sample'], batch['return']
        with torch.no_grad():
            Z = self.jepa.encode(X)

        B, S, _ = Z.shape
        ctx_len = self.rl_ctx_len
        max_steps = S - ctx_len
        steps = max_steps if self.rl_pred_steps is None else min(self.rl_pred_steps, max_steps)

        Z_p, Ret_p = Z[:, :ctx_len], Ret[:, :ctx_len]
        Act_p = torch.zeros((B, ctx_len, 1), device=Z.device, dtype=Z.dtype)
        vol = batch['vol'][:, ctx_len - 1:ctx_len]              # [B, 1, 1]

        Z_hat = ret_probs = None
        logits_l, act_l, idx_l, p_l = [], [], [], []

        for t in range(steps):
            with torch.no_grad():
                Zp1, ret_logits = self.jepa.predict(Z_p, Ret_p)
                rp = torch.softmax(ret_logits, dim=-1)

            # Row j of a causal pass never changes once written, so the first
            # step contributes the whole prompt and later steps one row each --
            # the buffer ends up identical to a single pass over the dream.
            if Z_hat is None:
                Z_hat, ret_probs = Zp1, rp
            else:
                Z_hat = torch.cat([Z_hat, Zp1[:, -1:]], dim=1)
                ret_probs = torch.cat([ret_probs, rp[:, -1:]], dim=1)

            lo = max(0, Z_hat.size(1) - self.backbone.max_len)
            cond = torch.cat((ret_probs[:, lo:], Act_p[:, lo:]), dim=-1)
            logits = self(Z_hat[:, lo:], cond)[:, -1:, :]

            src = logits if not detach_actions else logits.detach()
            new_act, new_idx = self.decode_actions(
                torch.softmax(src / self.act_temp, dim=-1), decode)

            # The market is sampled AFTER the allocation is chosen: the action
            # is taken before the return is realised, as on real data.
            with torch.no_grad():
                mprobs = torch.softmax(ret_logits[:, -1] / self.dream_temp, dim=-1)
                mid = torch.multinomial(mprobs, num_samples=1)          # [B, 1]
                r_sigma = self.jepa.return_bins[mid].unsqueeze(-1)      # [B, 1, 1]

            Z_p = truncate(torch.cat([Z_p, Zp1[:, -1:].detach()], dim=1),
                           self.jepa.predictor.max_len)
            Ret_p = truncate(torch.cat([Ret_p, r_sigma], dim=1),
                             self.jepa.predictor.max_len)
            Act_p = torch.cat([Act_p, new_act], dim=1)

            logits_l.append(logits)
            act_l.append(new_act)
            idx_l.append(new_idx)
            p_l.append(r_sigma * vol)

        return {
            'logits': torch.cat(logits_l, dim=1),
            'action': torch.cat(act_l, dim=1),
            'action_idx': torch.cat(idx_l, dim=1),
            'steps': steps,
            'act_path': Act_p[:, :ctx_len + steps - 1],
            'Z_hat': Z_hat,
            'ret_probs': ret_probs,
            'return_raw': torch.cat(p_l, dim=1),
        }

    def _realised(self, batch, roll, dream):
        """The return path the allocations actually earned, in real units."""
        if dream:
            return roll['return_raw'].squeeze(-1)
        lo = self.rl_ctx_len
        return batch['return_raw'][:, lo:lo + roll['steps']].squeeze(-1)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _drift(self, p):
        """Current estimate of the unconditional per-window drift.

        Estimated from the training stream only and applied only to the training
        REWARD -- `backtest`, and so every reported number, still runs on real
        returns in real units. This is reward shaping, not a look-ahead: nothing
        about the validation or test split enters the estimate.
        """
        if self.demean == 'none':
            return torch.zeros((), device=p.device, dtype=p.dtype)
        if self.demean == 'const':
            return torch.full((), self.drift_const, device=p.device, dtype=p.dtype)

        with torch.no_grad():
            m = p.mean().to(self.drift_ema.dtype)
            if int(self.drift_init) == 0:
                self.drift_ema.fill_(m)
                self.drift_init.fill_(1)
            # The PRE-update estimate is the one that gets used. Demeaning with
            # the current batch's own mean would strip the part of the drift the
            # policy could actually have predicted, not just the unconditional
            # part -- it would remove the signal along with the bias.
            cur = self.drift_ema.clone()
            if self.training:
                self.drift_ema.mul_(self.drift_beta).add_(m * (1.0 - self.drift_beta))
        return cur.to(device=p.device, dtype=p.dtype)

    def _rewards(self, actions, p):
        """Per-step reward in basis points of log growth, plus shaping.

        Returns `(rewards, dE_obj, dE_raw)`.

        `dE_raw` is the real, unshaped growth factor -- what the account would
        actually have earned -- and is what the episode logs report, so the
        shaping never leaks into a number anyone reads as a result. `dE_obj` is
        the shaped path the objective is built on: drift-removed (A) and, under
        `[rl.reward.benchmark] = 'bh'`, expressed relative to buy-and-hold (B).

        The sum of `rewards` over an episode is `[rl.reward.scale]` times log
        terminal equity -- log RELATIVE equity under the 'bh' benchmark -- which
        is why gamma = 1 is correct here rather than a hyperparameter. The path
        is built with the same leading flat position `_equity_bundle` prepends,
        so the reward and the backtest charge commission on identical turnover.
        """
        B = actions.size(0)
        a0 = torch.zeros((B, 1), device=actions.device, dtype=actions.dtype)
        x = torch.cat([a0, actions.squeeze(-1)], dim=1)               # [B, T+1]

        dE_raw = delta_equity(x, p, self.commission)                  # [B, T]
        p_eff = p - self._drift(p)
        dE = delta_equity(x, p_eff, self.commission)

        if self.benchmark == 'bh':
            # Fully long from the first step, so it pays commission exactly once
            # -- identical to the buy-and-hold leg of `_equity_bundle`, which is
            # what `val/excess_eq` is measured against.
            bh = torch.cat([a0, torch.ones_like(x[:, 1:])], dim=1)
            dE = dE / delta_equity(bh, p_eff, self.commission).clamp_min(1e-6)

        r = self.reward_scale * torch.log(dE.clamp_min(1e-6))

        # Both shaping terms are LINEAR in the reward's own units. A quadratic
        # cannot be made scale-coherent here -- squaring a bps quantity leaves
        # the coefficient carrying the units, which is how the old defaults came
        # to be off by three to six orders of magnitude.
        if self.downside_coef > 0:
            # Loss aversion: 1.0 makes a basis point lost count twice as much as
            # a basis point gained. Additive, so the return stays decomposable
            # per step, while pushing on the Sortino half of the definition of
            # done.
            r = r - self.downside_coef * torch.relu(-r)
        if self.reward_max_change is not None and self.turnover_penalty > 0:
            # Extra basis points charged per unit of allocation change beyond
            # the cap -- the same units as [data.actions.commission_value],
            # which is 5 bps.
            excess = torch.relu(torch.abs(torch.diff(x, dim=-1)) - self.reward_max_change)
            r = r - self.turnover_penalty * excess
        return r, dE, dE_raw

    def _update_ret_std(self, ret_tgt):
        """Track the spread of the GAE target so the critic stays O(1).

        The value target is log growth in bps summed over the episode: O(200) on
        this data. An MSE against that is not on speaking terms with a clipped
        surrogate of O(1), so `vf_coef` cannot be set sensibly for both. The
        critic therefore predicts `ret_tgt / ret_std` and `_score` multiplies the
        scale back in, leaving GAE and the reward in one consistent unit.
        """
        if not self.training:
            return
        with torch.no_grad():
            s = ret_tgt.std().clamp_min(1e-8).to(self.ret_std.dtype)
            if int(self.ret_init) == 0:
                self.ret_std.fill_(s)
                self.ret_init.fill_(1)
            else:
                self.ret_std.mul_(self.drift_beta).add_(s * (1.0 - self.drift_beta))

    def _gae(self, rewards, values):
        T = rewards.size(1)
        adv = torch.zeros_like(rewards)
        last = torch.zeros_like(rewards[:, 0])
        for t in reversed(range(T)):
            # The sequence ends here and the backtest does not charge for
            # closing the position, so the terminal bootstrap is 0.
            next_v = values[:, t + 1] if t < T - 1 else torch.zeros_like(last)
            delta = rewards[:, t] + self.gamma * next_v - values[:, t]
            last = delta + self.gamma * self.gae_lambda * last
            adv[:, t] = last
        return adv, adv + values

    # ------------------------------------------------------------------
    # Parallel replay of a collected episode
    # ------------------------------------------------------------------

    def _replay_inputs(self, roll):
        """Inputs that re-score a whole collected episode in one forward pass.

        The decision at step t read rows 0..ctx_len+t-1 of the world-model
        stream with the action path as its condition. Because the backbone is
        causal and the action path is fixed data once collected, running the
        union of those prefixes once and reading row ctx_len+t-1 gives exactly
        the logits step t saw.
        """
        steps = roll['steps']
        L = self.rl_ctx_len + steps - 1
        assert roll['Z_hat'] is not None, (
            'RL needs the one-pass world-model stream; the sequence is longer '
            'than jepa.predictor.max_len')
        assert L <= self.backbone.max_len, (
            f'rl.ctx_len + rl.pred_steps - 1 = {L} exceeds actor.backbone.max_len '
            f'({self.backbone.max_len}); the replay pass would be truncated')
        Z_in = roll['Z_hat'][:, :L]
        cond = torch.cat((roll['ret_probs'][:, :L], roll['act_path'][:, :L]), dim=-1)
        return Z_in, cond, steps

    def _score(self, Z_in, cond, steps):
        """Action logits and the critic's NORMALISED value.

        The second return is in units of `ret_std`, not of the reward -- see
        `_update_ret_std`. Callers multiply by `ret_std` to put it back on the
        reward's scale for GAE, and train the head against the normalised
        target. Zero-init still means an exact zero value at step 0.
        """
        h = self.backbone(Z_in, cond)
        lo = self.rl_ctx_len - 1
        h = h[:, lo:lo + steps]
        logits = self.actor_head(h)
        v_norm = self.critic_head(h.detach() if self.detach_trunk else h).squeeze(-1)
        return logits, v_norm

    def _ref_logits(self, Z_in, cond, steps):
        with torch.no_grad():
            h = self.ref_backbone(Z_in, cond)
            lo = self.rl_ctx_len - 1
            return self.ref_head(h[:, lo:lo + steps])

    @staticmethod
    def _kl(logits, ref_logits):
        """KL(pi || pi_ref), exact over the categorical -- no sampling needed."""
        logp = F.log_softmax(logits, dim=-1)
        ref_logp = F.log_softmax(ref_logits, dim=-1)
        return (logp.exp() * (logp - ref_logp)).sum(-1).mean()

    @staticmethod
    def _entropy(logits):
        logp = F.log_softmax(logits, dim=-1)
        return -(logp.exp() * logp).sum(-1).mean()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        if self.objective == 'analytic':
            self._analytic_step(batch)
        else:
            self._ppo_step(batch)
        # One scheduler step per collected batch, not per inner update, so the
        # cosine spans trainer.estimated_stepping_batches as it does elsewhere.
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

    def _log_episode(self, actions, dE_raw, dE_obj=None, tag='rl'):
        """Episode diagnostics, reported on the REAL growth path.

        `dE_raw` is unshaped, so end_equity and sortino here stay comparable to
        the backtest no matter what the reward is doing. `mean_alloc` is the
        collapse detector: a policy that has gone all-in long reads mean_alloc
        ~ +1, abs_alloc ~ 1 and turnover ~ 0.
        """
        with torch.no_grad():
            eq = dE_raw.clamp_min(1e-6).log().sum(dim=-1).exp()
            a = actions.squeeze(-1)
            turn = (torch.abs(torch.diff(a, dim=-1)).mean() if a.size(1) > 1
                    else torch.zeros((), device=a.device))
            d = {
                f'{tag}/end_equity': eq.mean(),
                f'{tag}/sortino': sortino_ratio(dE_raw).mean(),
                f'{tag}/turnover': turn,
                f'{tag}/mean_alloc': a.mean(),
                f'{tag}/abs_alloc': a.abs().mean(),
                f'{tag}/drift_ema': self.drift_ema,
            }
            if dE_obj is not None:
                # Under the 'bh' benchmark this is equity RELATIVE to
                # buy-and-hold, so 1.0 means the policy exactly matched it.
                d[f'{tag}/obj_equity'] = (
                    dE_obj.clamp_min(1e-6).log().sum(dim=-1).exp().mean())
            self.log_dict(d, on_step=True, on_epoch=False)

    def _ppo_step(self, batch):
        opt = self.optimizers()
        dream = self._use_dream()

        with torch.no_grad():
            roll = self._rollout(batch, decode='sample', detach_actions=True, dream=dream)
            actions, act_idx = roll['action'], roll['action_idx']
            p = self._realised(batch, roll, dream)
            rewards, dE, dE_raw = self._rewards(actions, p)

            Z_in, cond, steps = self._replay_inputs(roll)
            # The collection logits ARE the behaviour policy's logits (dropout
            # is off and the replay is exact), so the ratio starts at exactly 1.
            old_logp = torch.log_softmax(roll['logits'], dim=-1).gather(
                -1, act_idx.long()).squeeze(-1)

            _, v_norm = self._score(Z_in, cond, steps)
            values = v_norm * self.ret_std
            adv, ret_tgt = self._gae(rewards, values)

            # (E) Centre the advantage per timestep across the batch. The critic
            # is zero-initialised and true V really is ~0 here, so it supplies
            # almost no baseline; the batch mean at each t is a free time-varying
            # one that removes the systematic component of the reward the action
            # had no say in -- the entry commission everyone pays at t = 0, and
            # whatever drift the EMA in (A) has not caught up with. It is biased
            # by the sample's own O(1/B) contribution, which is negligible at
            # [rl.training.batch_size] and worth it for the variance.
            if self.center_adv and adv.size(0) > 1:
                adv = adv - adv.mean(dim=0, keepdim=True)

            # Keep the critic's target O(1) whatever [rl.reward.scale] is.
            self._update_ret_std(ret_tgt)
            v_tgt = ret_tgt / self.ret_std

            ref_logits = (self._ref_logits(Z_in, cond, steps)
                          if self.ref_backbone is not None else None)

        self._log_episode(actions, dE_raw, dE)

        B = Z_in.size(0)
        mb_size = self.minibatch or B
        stop = False
        for _ in range(self.ppo_epochs):
            if stop:
                break
            perm = torch.randperm(B, device=Z_in.device)
            for mb in perm.split(mb_size):
                logits, v = self._score(Z_in[mb], cond[mb], steps)
                logp = torch.log_softmax(logits, dim=-1).gather(
                    -1, act_idx[mb].long()).squeeze(-1)

                ratio = torch.exp(logp - old_logp[mb])
                a = adv[mb]
                if self.norm_adv:
                    a = (a - a.mean()) / (a.std() + 1e-8)

                pg = -torch.min(ratio * a,
                                ratio.clamp(1 - self.ppo_clip, 1 + self.ppo_clip) * a).mean()
                # Both sides normalised, so this is O(1) and vf_coef is legible.
                vf = F.mse_loss(v, v_tgt[mb])
                ent = self._entropy(logits)
                kl = (self._kl(logits, ref_logits[mb])
                      if ref_logits is not None else torch.zeros((), device=logits.device))

                if int(self.updates) < self.critic_warmup:
                    # Let the value head leave its zero init before its
                    # advantage estimates are allowed to move the policy.
                    loss = self.vf_coef * vf
                else:
                    loss = pg + self.vf_coef * vf - self.ent_coef * ent + self.kl_coef * kl

                opt.zero_grad()
                self.manual_backward(loss)
                self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm='norm')
                opt.step()
                self.updates += 1

                with torch.no_grad():
                    approx_kl = (old_logp[mb] - logp).mean()
                    clipfrac = ((ratio - 1).abs() > self.ppo_clip).float().mean()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    stop = True
                    break

        self.log_dict({
            'rl/policy_loss': pg.detach(),
            'rl/value_loss': vf.detach(),
            'rl/entropy': ent.detach(),
            'rl/kl_ref': kl.detach(),
            'rl/approx_kl': approx_kl,
            'rl/clipfrac': clipfrac,
            'rl/adv_std': adv.std(),
            'rl/value_mean': values.mean(),
            'rl/ret_std': self.ret_std,
            'rl/reward': rewards.mean(),
            'rl/dream': float(dream),
        }, on_step=True, on_epoch=False)

    def _analytic_step(self, batch):
        """Backpropagate the objective through the actions themselves.

        Valid because the transition is action-independent and the reward is a
        differentiable function of the allocation: there is no unknown Jacobian
        to estimate, so the gradient is exact rather than sampled. The decode
        must be 'expected' -- sampling a bin would cut the path.
        """
        opt = self.optimizers()
        dream = self._use_dream()

        roll = self._rollout(batch, decode='expected', detach_actions=False, dream=dream)
        actions = roll['action']
        p = self._realised(batch, roll, dream)
        rewards, dE, dE_raw = self._rewards(actions, p)

        if self.analytic_objective == 'sortino':
            # On the SHAPED path, so this is a Sortino on excess growth when the
            # benchmark is on. Dimensionless and per-step: O(0.01) here.
            obj = sortino_ratio(dE)
        else:
            # Mean, not sum: log terminal equity per step, in basis points, so
            # the objective does not silently change scale with [rl.pred_steps]
            # and the KL coefficient below stays meaningful across horizons.
            obj = rewards.mean(dim=1)
        obj = self.obj_scale * obj
        loss = -obj.mean()

        logits = roll['logits']
        ent = self._entropy(logits)
        loss = loss - self.ent_coef * ent

        kl = torch.zeros((), device=logits.device)
        if self.ref_backbone is not None:
            with torch.no_grad():
                Z_in, cond, steps = self._replay_inputs(roll)
                ref_logits = self._ref_logits(Z_in, cond, steps)
            kl = self._kl(logits, ref_logits)
            loss = loss + self.kl_coef * kl

        opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm='norm')
        opt.step()
        self.updates += 1

        self._log_episode(actions.detach(), dE_raw.detach(), dE.detach())
        self.log_dict({
            'rl/analytic_loss': loss.detach(),
            'rl/analytic_obj': obj.detach().mean(),
            'rl/entropy': ent.detach(),
            'rl/kl_ref': kl.detach(),
            'rl/reward': rewards.detach().mean(),
            'rl/dream': float(dream),
        }, on_step=True, on_epoch=False)

    # ------------------------------------------------------------------
    # Validation -- identical protocol to Actor / ActorAR
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        b = self.backtest(batch, pred_steps=self.val_pred_steps)

        # Behaviour-cloning CE against the oracle, as a DRIFT diagnostic only.
        # It is expected to get worse: if RL is finding anything the clairvoyant
        # labels do not contain, the policy has to move away from them.
        act_logits = self._teacher_forced_logits(batch)
        bc_ce = self.CrossEntropyLoss(
            act_logits.reshape(-1, act_logits.size(-1)),
            batch['action_target'][:, 1:].reshape(-1).long(),
        )

        a = b['action'].squeeze(-1)
        dE = delta_equity(
            torch.cat([torch.zeros_like(a[:, :1]), a], dim=1),
            b['return_raw'].squeeze(-1), self.commission)

        e = b['end_equity'].clamp_min(1e-6)
        bh = b['bh_end_equity'].clamp_min(1e-6)

        self.log_dict({
            'val/mean_eq': e.mean(),
            'val/opt_eq': b['opt_end_equity'].mean(),
            'val/bh_eq': bh.mean(),
            # (C) The checkpoint and early-stopping monitor. mean_eq cannot
            # separate skill from drift: a policy that outputs +1 everywhere IS
            # buy-and-hold and scores exactly bh_eq, so selecting on mean_eq
            # rewards the collapse it is supposed to catch. This is
            # log(E_actor / E_bh) -- 0 for that policy, positive only for timing
            # -- which is what the definition of done actually asks for.
            'val/excess_eq': (e.log() - bh.log()).mean(),
            # A flat policy never trades and never earns: exactly 1.0. Logged
            # because the definition of done names it as a baseline.
            'val/flat_eq': torch.ones((), device=a.device),
            'val/sortino': sortino_ratio(dE).mean(),
            'val/win_rate': (b['end_equity'] > 1.0).float().mean(),
            'val/turnover': torch.abs(torch.diff(a, dim=-1)).mean(),
            # The collapse detector: +1 means the policy has become long-only.
            'val/mean_alloc': a.mean(),
            'val/abs_alloc': a.abs().mean(),
            'val/bc_ce': bc_ce,
        }, on_step=False, on_epoch=True, prog_bar=True)
