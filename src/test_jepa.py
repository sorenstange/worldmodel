"""Evaluate the world model on the held-out test split.

Produces, in ./figs/dreams/:
  horizon_skill.png   latent skill and return NLL vs rollout horizon
  calibration.png     reliability diagram + PIT histogram
  confusion.png       predicted vs realised return bin
  return_dist.png     dreamed vs realised return distribution
  dream{i}.png        sample open-loop rollouts against the true path

and a metrics table on stdout plus ./figs/dreams/metrics.json.

Every headline number is quoted against a baseline. A raw latent MSE or a raw
NLL is unreadable on its own -- what matters is whether the model beats
'assume nothing changes' and 'predict the marginal distribution'.
"""

import json
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import viz
from data import CryptoDataset
from jepa import JEPA
from metrics import (bits_gained, crps, directional_accuracy, effective_rank,
                     expected_values, format_table, marginal_nll, nll, pit,
                     pit_uniformity, reliability, rps, top_k_accuracy)
from util import set_logger
from viz import plt

BATCH_SIZE = 64
MAX_BATCHES = 40          # cap the pass; the test split is large
HORIZON = 32
NUM_PLOTS = 8
RET_TEMPERATURE = 1.0


def collect(model, loader, device, horizon, logger):
    """One pass: teacher-forced one-step stats, plus a rollout from a fixed
    context so per-horizon degradation is measurable."""
    probs, targets, latents = [], [], []
    roll_mse, pers_mse, roll_nll = [], [], []
    keep = None

    for i, batch in enumerate(loader):
        if i >= MAX_BATCHES:
            break
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            Z = model.encode(batch['sample'])
            _, logits = model.predict(Z[:, :-1], batch['return'][:, :-1])

        probs.append(torch.softmax(logits, dim=-1).reshape(-1, logits.size(-1)).cpu().numpy())
        targets.append(batch['return_target'][:, 1:].reshape(-1).cpu().numpy())
        latents.append(Z.cpu().numpy())

        # Deterministic rollout (expected-return feedback) from a context that
        # leaves `horizon` steps of ground truth to score against.
        c = Z.size(1) - horizon
        if c >= 1:
            with torch.no_grad():
                pred_Z, pred_logits = model.rollout(Z[:, :c], batch['return'][:, :c], horizon)

            true_Z = Z[:, c:c + horizon]
            # Persistence: nothing changes after the last observed latent.
            persistence = Z[:, c - 1:c].expand_as(true_Z)

            roll_mse.append(((pred_Z - true_Z) ** 2).mean(dim=(0, 2)).cpu().numpy())
            pers_mse.append(((persistence - true_Z) ** 2).mean(dim=(0, 2)).cpu().numpy())

            tgt = batch['return_target'][:, c:c + horizon].squeeze(-1)
            lp = torch.log_softmax(pred_logits, dim=-1)
            roll_nll.append((-lp.gather(-1, tgt.unsqueeze(-1).long()).squeeze(-1))
                            .mean(dim=0).cpu().numpy())

        if keep is None:
            keep = (batch, Z)

    logger.info(f'Scored {len(probs)} batches.')
    return {
        'probs': np.concatenate(probs),
        'targets': np.concatenate(targets).astype(int),
        'latents': np.concatenate(latents),
        'roll_mse': np.mean(roll_mse, axis=0) if roll_mse else None,
        'pers_mse': np.mean(pers_mse, axis=0) if pers_mse else None,
        'roll_nll': np.mean(roll_nll, axis=0) if roll_nll else None,
        'keep': keep,
    }


def plot_horizon_skill(d, horizon, folder):
    """Two panels, never two y-axes on one panel: the scales are unrelated."""
    h = np.arange(1, horizon + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    skill = 1.0 - d['roll_mse'] / np.maximum(d['pers_mse'], 1e-12)
    ax1.axhline(0.0, color=viz.REFERENCE, linewidth=1.2, linestyle='--')
    ax1.annotate('persistence baseline', xy=(h[-1], 0.0), xytext=(-4, 6),
                 textcoords='offset points', ha='right', color=viz.INK_SOFT, fontsize=9)
    ax1.plot(h, skill, color=viz.C_ACTOR)
    ax1.set_title('Latent skill score vs horizon')
    ax1.set_xlabel('Rollout step (15m)')
    ax1.set_ylabel('1 - MSE / MSE(persistence)')

    ax2.plot(h, d['roll_nll'], color=viz.C_ACTOR, label='Rollout')
    ax2.axhline(d['marginal_nll'], color=viz.REFERENCE, linewidth=1.2, linestyle='--')
    ax2.annotate('marginal baseline', xy=(h[-1], d['marginal_nll']), xytext=(-4, 6),
                 textcoords='offset points', ha='right', color=viz.INK_SOFT, fontsize=9)
    ax2.set_title('Return NLL vs horizon')
    ax2.set_xlabel('Rollout step (15m)')
    ax2.set_ylabel('NLL (nats)')

    viz.save(fig, f'{folder}/horizon_skill.png')


def plot_calibration(d, folder):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    xs, ys, ws, ece = d['reliability']
    ax1.plot([0, 1], [0, 1], color=viz.REFERENCE, linestyle='--', linewidth=1.2)
    ax1.annotate('perfect calibration', xy=(0.98, 0.98), ha='right', va='top',
                 color=viz.INK_SOFT, fontsize=9)
    ax1.plot(xs, ys, color=viz.C_ACTOR, linewidth=2, zorder=3)
    # Marker area tracks how many predictions landed in each confidence bin, so a
    # dramatic-looking excursion built on a handful of samples reads as the small
    # thing it is rather than as real miscalibration.
    ax1.scatter(xs, ys, s=20 + 260 * ws / max(ws.max(), 1), color=viz.C_ACTOR,
                edgecolor=viz.SURFACE, linewidth=1.2, zorder=4)
    ax1.set_title(f'Reliability (ECE {ece:.4f})')
    ax1.set_xlabel('Predicted confidence')
    ax1.set_ylabel('Observed accuracy')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.hist(d['pit'], bins=20, range=(0, 1), color=viz.C_ACTOR, alpha=0.85,
             edgecolor=viz.SURFACE, linewidth=1.2)
    ax2.axhline(len(d['pit']) / 20, color=viz.REFERENCE, linestyle='--', linewidth=1.2)
    ax2.annotate('uniform', xy=(0.98, len(d['pit']) / 20), xytext=(0, 6),
                 textcoords='offset points', ha='right', color=viz.INK_SOFT, fontsize=9)
    ax2.set_title(f"PIT histogram (chi2 {d['pit_chi2']:.1f})")
    ax2.set_xlabel('PIT value')
    ax2.set_ylabel('Count')

    viz.save(fig, f'{folder}/calibration.png')


def plot_confusion(d, cfg, folder):
    K = cfg['data']['returns']['num_bins']
    lo, hi = cfg['data']['returns']['min_value'], cfg['data']['returns']['max_value']

    pred = d['probs'].argmax(axis=1)
    H, _, _ = np.histogram2d(d['targets'], pred, bins=[K, K], range=[[0, K], [0, K]])
    H = H / max(H.sum(), 1)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    viz.heatmap(ax, H, extent=[lo, hi, lo, hi], label='Fraction of predictions')
    ax.plot([lo, hi], [lo, hi], color=viz.REFERENCE, linestyle='--', linewidth=1.2)
    ax.set_title('Predicted vs realised return bin')
    ax.set_xlabel('Predicted (sigma)')
    ax.set_ylabel('Realised (sigma)')
    viz.save(fig, f'{folder}/confusion.png')


def plot_return_dist(true_norm, dream_norm, folder):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bins = np.linspace(-5, 5, 61)
    ax.hist(true_norm, bins=bins, density=True, color=viz.C_MARKET, alpha=0.55,
            label='Realised', edgecolor=viz.SURFACE, linewidth=0.8)
    ax.hist(dream_norm, bins=bins, density=True, histtype='step',
            color=viz.C_ACTOR, linewidth=2, label='Dreamed')
    ax.set_title('Dreamed vs realised return distribution')
    ax.set_xlabel('Return (sigma)')
    ax.set_ylabel('Density')
    ax.legend()
    viz.save(fig, f'{folder}/return_dist.png')


def plot_dream(i, t_steps, d, cfg, path, horizon):
    lo, hi = cfg['data']['returns']['min_value'], cfg['data']['returns']['max_value']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True)

    ax1.axhline(0.0, color=viz.REFERENCE, linewidth=1.0)
    ax1.plot(t_steps, 100 * d['t_cumprod'][i], color=viz.C_MARKET, label='Realised')
    ax1.plot(t_steps, 100 * d['d_cumprod'][i], color=viz.C_ACTOR, linestyle=':', label='Dreamed')
    ax1.set_title('Open-loop rollout vs realised path')
    ax1.set_ylabel('Cumulative return (%)')
    ax1.legend()

    viz.heatmap(ax2, d['d_ret_probs'][i].T, extent=[0.5, horizon + 0.5, lo, hi],
                label='Probability')
    ax2.scatter(t_steps, d['t_ret_norm'][i], s=34, color=viz.C_MARKET,
                edgecolor=viz.SURFACE, linewidth=1.0, label='Realised', zorder=5)
    ax2.scatter(t_steps, d['d_ret_norm'][i], s=40, marker='x', color=viz.C_ORACLE,
                linewidth=1.8, label='Sampled', zorder=6)
    ax2.set_title('Predicted return distribution')
    ax2.set_xlabel('Rollout step (15m)')
    ax2.set_ylabel('Return (sigma)')
    ax2.legend(loc='upper left')

    viz.save(fig, path)


def main():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.to_container(OmegaConf.load('./config.yaml'), resolve=True)
    logger = set_logger(cfg)
    viz.use_style()

    folder = './figs/dreams'
    os.makedirs(folder, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = JEPA.load_from_checkpoint(f"./models/{cfg['jepa']['name']}/best.ckpt", cfg=cfg)
    model.eval().to(device)

    loader = DataLoader(CryptoDataset(cfg, mode='test'), batch_size=BATCH_SIZE, shuffle=False)
    d = collect(model, loader, device, HORIZON, logger)

    K = cfg['data']['returns']['num_bins']
    bin_values = model.return_bins.cpu().numpy()
    bin_width = float(bin_values[1] - bin_values[0])
    rng = np.random.default_rng(0)

    d['marginal_nll'] = marginal_nll(d['targets'], K)
    d['reliability'] = reliability(d['probs'], d['targets'])
    d['pit'] = pit(d['probs'], d['targets'], rng)
    d['pit_chi2'] = pit_uniformity(d['pit'])

    true_norm = bin_values[d['targets']]
    stats = {
        'n_predictions': int(len(d['targets'])),
        'nll': nll(d['probs'], d['targets']),
        'nll_marginal': d['marginal_nll'],
        'bits_gained': bits_gained(d['probs'], d['targets'], K),
        'top1_acc': top_k_accuracy(d['probs'], d['targets'], 1),
        'top5_acc': top_k_accuracy(d['probs'], d['targets'], 5),
        'rps': rps(d['probs'], d['targets']),
        'crps_sigma': crps(d['probs'], d['targets'], bin_width),
        'directional_acc': directional_accuracy(d['probs'], bin_values, true_norm),
        'ece': d['reliability'][3],
        'pit_chi2': d['pit_chi2'],
        'latent_erank': effective_rank(d['latents']),
        'latent_dim': int(d['latents'].shape[-1]),
        'latent_std': float(d['latents'].reshape(-1, d['latents'].shape[-1]).std(axis=0).mean()),
        'pred_mae_sigma': float(np.mean(np.abs(expected_values(d['probs'], bin_values) - true_norm))),
    }

    if d['roll_mse'] is not None:
        def skill_at(h):
            """1 - MSE / MSE(persistence) at horizon h (1-indexed)."""
            i = min(h, len(d['roll_mse'])) - 1
            return float(1 - d['roll_mse'][i] / max(d['pers_mse'][i], 1e-12))

        stats['latent_skill_h1'] = skill_at(1)
        stats['latent_skill_h8'] = skill_at(8)
        stats['latent_skill_hmax'] = skill_at(len(d['roll_mse']))
        stats['rollout_nll_h1'] = float(d['roll_nll'][0])
        stats['rollout_nll_hmax'] = float(d['roll_nll'][-1])
        plot_horizon_skill(d, HORIZON, folder)

    plot_calibration(d, folder)
    plot_confusion(d, cfg, folder)

    # Sampled rollouts for the qualitative figures and the distribution check.
    batch, Z_true = d['keep']
    start = Z_true.size(1) - HORIZON - 1
    with torch.no_grad():
        d_states, d_ret, d_ret_probs, _ = model.dream(
            Z_true[:, :start + 1], batch['return'][:, :start + 1],
            horizon=HORIZON, ret_temperature=RET_TEMPERATURE)

    vol = batch['vol'][:, start + 1:].squeeze(-1)
    t_raw = batch['return_raw'][:, start + 1:].squeeze(-1)
    d_raw = d_ret.squeeze(-1) * vol

    plots = {
        't_ret_norm': batch['return'][:, start + 1:].squeeze(-1).cpu().numpy(),
        'd_ret_norm': d_ret.squeeze(-1).cpu().numpy(),
        'd_ret_probs': d_ret_probs.cpu().numpy(),
        't_cumprod': (torch.cumprod(1 + t_raw, dim=1) - 1).cpu().numpy(),
        'd_cumprod': (torch.cumprod(1 + d_raw, dim=1) - 1).cpu().numpy(),
    }
    t_states = Z_true[:, start + 1:].cpu().numpy()
    d_states = d_states.cpu().numpy()

    plot_return_dist(plots['t_ret_norm'].ravel(), plots['d_ret_norm'].ravel(), folder)

    stats['dream_std_ratio'] = float(plots['d_ret_norm'].std() / (plots['t_ret_norm'].std() + 1e-12))

    t_steps = np.arange(1, HORIZON + 1)
    for i in range(min(NUM_PLOTS, d_states.shape[0])):
        plot_dream(i, t_steps, plots, cfg, f'{folder}/dream{i + 1}.png', HORIZON)
    stats['dream_latent_mse'] = float(np.mean((t_states - d_states) ** 2))

    rows = [{'metric': k, 'value': v} for k, v in stats.items()]
    logger.info('\n' + format_table(rows, ['metric', 'value'], ['Metric', 'Value'], '{:.5f}'))

    with open(f'{folder}/metrics.json', 'w') as f:
        json.dump(stats, f, indent=2, default=float)

    logger.info('')
    logger.info(f"Bits gained over the marginal baseline: {stats['bits_gained']:+.4f} "
                f"(<= 0 means the model has learned nothing about the conditional)")
    logger.info(f"Directional accuracy: {stats['directional_acc']:.4f} (0.5 = coin flip)")
    logger.info(f"Latent effective rank: {stats['latent_erank']:.2f} / {stats['latent_dim']} "
                f"(near 1 means collapse)")
    logger.info(f"Figures and metrics.json written to '{folder}'")


if __name__ == '__main__':
    main()
