"""Evaluation metrics for the world model and the trading policy.

Pure numpy on detached arrays, so these are testable without a GPU and reusable
from both test scripts. Nothing here mutates its inputs.

Two conventions used throughout:
  * `probs`      -- [N, K] rows summing to 1 over K ordered bins.
  * `equity`     -- [N, T] cumulative equity per sequence, implicitly starting
                    from 1.0 at t = -1.
"""

import numpy as np

# 15m steps in a 365-day year. Sequences are disjoint 16h windows rather than one
# continuous track record, so annualised figures pool step returns across
# sequences -- report them as indicative, not as a live track record.
STEPS_PER_YEAR = 365 * 24 * 4
EPS = 1e-12


# ----------------------------------------------------------------------
# Probabilistic forecast quality
# ----------------------------------------------------------------------

def nll(probs, target_idx):
    """Mean negative log-likelihood in nats."""
    p = probs[np.arange(len(target_idx)), target_idx]
    return float(-np.mean(np.log(p + EPS)))


def marginal_nll(target_idx, num_bins):
    """NLL of the best constant predictor: the empirical marginal.

    This is the honest 'no skill' reference. A model that cannot beat it has
    learned nothing about the conditional distribution.
    """
    counts = np.bincount(target_idx, minlength=num_bins).astype(float)
    marginal = counts / max(counts.sum(), 1.0)
    return float(-np.mean(np.log(marginal[target_idx] + EPS)))


def bits_gained(probs, target_idx, num_bins):
    """How many bits per prediction the model buys over the marginal."""
    return float((marginal_nll(target_idx, num_bins) - nll(probs, target_idx)) / np.log(2.0))


def top_k_accuracy(probs, target_idx, k=1):
    if k == 1:
        return float(np.mean(probs.argmax(axis=1) == target_idx))
    top = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float(np.mean((top == target_idx[:, None]).any(axis=1)))


def rps(probs, target_idx):
    """Ranked probability score -- the ordinal analogue of the Brier score.

    Unlike accuracy this rewards being close: putting mass one bin away costs
    far less than putting it at the far end. Lower is better.
    """
    cdf = np.cumsum(probs, axis=1)
    K = probs.shape[1]
    step = (np.arange(K)[None, :] >= target_idx[:, None]).astype(float)
    return float(np.mean(np.sum((cdf - step) ** 2, axis=1)))


def crps(probs, target_idx, bin_width):
    """RPS carried into the target's own units (sigma, here)."""
    return rps(probs, target_idx) * bin_width


def pit(probs, target_idx, rng):
    """Randomised probability integral transform.

    For a calibrated forecast these are Uniform(0, 1). The randomisation is what
    makes the transform valid for a discrete distribution -- without it the
    histogram is spuriously lumpy.
    """
    cdf = np.cumsum(probs, axis=1)
    rows = np.arange(len(target_idx))
    upper = cdf[rows, target_idx]
    lower = upper - probs[rows, target_idx]
    return lower + rng.random(len(target_idx)) * (upper - lower)


def pit_uniformity(u, n_bins=20):
    """Chi-square statistic of the PIT histogram against uniform.

    Reported alongside the plot so 'looks flat' is backed by a number.
    """
    counts, _ = np.histogram(u, bins=n_bins, range=(0.0, 1.0))
    expected = len(u) / n_bins
    return float(np.sum((counts - expected) ** 2) / max(expected, EPS))


def reliability(probs, target_idx, n_bins=12):
    """Confidence vs accuracy for the top-1 prediction, plus ECE."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == target_idx).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)

    xs, ys, ws = [], [], []
    ece = 0.0
    for b in range(n_bins):
        m = which == b
        if not m.any():
            continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        xs.append(c)
        ys.append(a)
        ws.append(int(m.sum()))
        ece += (m.sum() / len(conf)) * abs(a - c)

    return np.array(xs), np.array(ys), np.array(ws), float(ece)


def directional_accuracy(probs, bin_values, true_values):
    """Did the model call the sign of the move?

    The single most trading-relevant thing a return head can get right. Ties in
    the true value are dropped rather than scored.
    """
    p_up = probs[:, bin_values > 0].sum(axis=1)
    p_down = probs[:, bin_values < 0].sum(axis=1)
    pred_up = p_up > p_down

    mask = true_values != 0
    if not mask.any():
        return float('nan')
    return float(np.mean(pred_up[mask] == (true_values[mask] > 0)))


def expected_values(probs, bin_values):
    return probs @ bin_values


# ----------------------------------------------------------------------
# Latent dynamics
# ----------------------------------------------------------------------

def latent_r2(pred, true, baseline):
    """1 - MSE(pred) / MSE(baseline).

    A raw latent MSE is unreadable on its own -- it depends entirely on the scale
    the encoder happens to settle at. Against a persistence baseline it becomes
    a skill score: > 0 beats 'assume nothing changes', <= 0 does not.
    """
    num = np.mean((pred - true) ** 2)
    den = np.mean((baseline - true) ** 2)
    return float(1.0 - num / (den + EPS))


def effective_rank(Z):
    """exp(entropy of the normalised eigenvalue spectrum) of the latent
    covariance. Near D is healthy; near 1 means collapse."""
    Zf = Z.reshape(-1, Z.shape[-1]).astype(np.float64)
    Zc = Zf - Zf.mean(axis=0, keepdims=True)
    cov = (Zc.T @ Zc) / max(len(Zc) - 1, 1)
    ev = np.clip(np.linalg.eigvalsh(cov), 0, None)
    total = ev.sum()
    if total <= 0:
        return 0.0
    p = ev / total
    return float(np.exp(-np.sum(p * np.log(p + EPS))))


# ----------------------------------------------------------------------
# Trading performance
# ----------------------------------------------------------------------

def step_returns(equity):
    """Per-step simple returns of an equity curve that starts at 1.0."""
    prepended = np.concatenate([np.ones((equity.shape[0], 1)), equity], axis=1)
    return prepended[:, 1:] / np.maximum(prepended[:, :-1], EPS) - 1.0


def max_drawdown(equity):
    """Worst peak-to-trough fraction per sequence."""
    prepended = np.concatenate([np.ones((equity.shape[0], 1)), equity], axis=1)
    peak = np.maximum.accumulate(prepended, axis=1)
    return np.max(1.0 - prepended / np.maximum(peak, EPS), axis=1)


def sharpe(r):
    return float(np.mean(r) / (np.std(r) + EPS))


def sortino(r):
    downside = np.sqrt(np.mean(np.clip(-r, 0, None) ** 2))
    return float(np.mean(r) / (downside + EPS))


def profit_factor(r):
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return float(gains / (losses + EPS))


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Percentile CI for the mean, resampling SEQUENCES.

    Resampling individual steps would ignore within-sequence autocorrelation and
    give a CI that is far too tight.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def turnover(actions):
    """Mean |change in allocation| per step, opening from flat."""
    prepended = np.concatenate([np.zeros((actions.shape[0], 1)), actions], axis=1)
    return float(np.mean(np.abs(np.diff(prepended, axis=1))))


def summarize(equity, actions=None, commission=0.0, label=''):
    """Everything worth quoting about one strategy, as a flat dict."""
    end = equity[:, -1]
    r = step_returns(equity).ravel()
    dd = max_drawdown(equity)
    ann = np.sqrt(STEPS_PER_YEAR)

    lo, hi = bootstrap_ci(end - 1.0)
    n = len(end)
    tstat = float(np.mean(end - 1.0) / (np.std(end - 1.0, ddof=1) / np.sqrt(n) + EPS)) if n > 1 else float('nan')

    out = {
        'label': label,
        'sequences': int(n),
        'roi_mean': float(np.mean(end) - 1.0),
        'roi_median': float(np.median(end) - 1.0),
        'roi_ci_lo': lo,
        'roi_ci_hi': hi,
        't_stat': tstat,
        'sharpe_step': sharpe(r),
        'sharpe_ann': sharpe(r) * ann,
        'sortino_step': sortino(r),
        'sortino_ann': sortino(r) * ann,
        'max_dd_mean': float(np.mean(dd)),
        'max_dd_worst': float(np.max(dd)),
        'win_rate': float(np.mean(end > 1.0)),
        'profit_factor': profit_factor(r),
    }
    out['calmar'] = float(out['roi_mean'] / (out['max_dd_mean'] + EPS))

    if actions is not None:
        a = actions.reshape(actions.shape[0], -1)
        turn = turnover(a)
        out.update({
            'turnover': turn,
            'cost_drag': float(commission * turn * a.shape[1]),
            'exposure_abs': float(np.mean(np.abs(a))),
            'frac_long': float(np.mean(a > 0.05)),
            'frac_short': float(np.mean(a < -0.05)),
            'frac_flat': float(np.mean(np.abs(a) <= 0.05)),
        })
    return out


def format_table(rows, columns, headers=None, floatfmt='{:.4f}'):
    """Fixed-width text table, so results paste cleanly into a log or an issue."""
    headers = headers or columns
    cells = [[str(h) for h in headers]]
    for row in rows:
        line = []
        for c in columns:
            v = row.get(c, '')
            line.append(floatfmt.format(v) if isinstance(v, float) else str(v))
        cells.append(line)

    widths = [max(len(r[i]) for r in cells) for i in range(len(columns))]
    sep = '-+-'.join('-' * w for w in widths)

    out = [' | '.join(c.ljust(widths[i]) for i, c in enumerate(cells[0])), sep]
    out += [' | '.join(c.ljust(widths[i]) for i, c in enumerate(r)) for r in cells[1:]]
    return '\n'.join(out)
