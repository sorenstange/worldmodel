"""Evaluate the trading policy on the held-out test split.

Produces, in ./figs/backtest/:
  equity_summary.png   median equity + IQR band, actor vs oracle vs buy-and-hold
  outcomes.png         end-equity distribution and drawdown distribution
  allocations.png      allocation distributions and actor-vs-oracle agreement
  per_symbol.png       ROI by symbol, actor vs buy-and-hold
  backtest{i}.png      individual sequences
and a metrics table on stdout plus ./figs/backtest/metrics.json.

Every number is quoted against buy-and-hold and the clairvoyant oracle, with a
bootstrap CI and a t-stat on the mean, because a positive mean ROI over a few
hundred correlated 16h windows is not on its own evidence of anything.
"""

import argparse
import json
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import viz
from actor import Actor, ActorAR
from data import CryptoDataset
from jepa import JEPA
from metrics import STEPS_PER_YEAR, format_table, max_drawdown, summarize
from util import set_logger
from viz import plt

NUM_PLOT_SEQS = 8


def run_backtest(model, loader, device, decode, act_temp, logger):
    out = {k: [] for k in ('equity', 'opt_equity', 'bh_equity', 'end_equity',
                           'opt_end_equity', 'bh_end_equity', 'action',
                           'opt_action', 'return_raw', 'symbol_id')}
    first = None

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            b = model.backtest(batch, act_temp=act_temp, decode=decode)
        b = {k: v.detach().cpu().numpy() for k, v in b.items()}

        for k in out:
            if k == 'symbol_id':
                out[k].append(batch['symbol_id'].cpu().numpy())
            else:
                out[k].append(b[k])
        if first is None:
            first = b

    logger.info(f'Backtested {sum(len(x) for x in out["end_equity"]):,} sequences.')
    return {k: np.concatenate(v) for k, v in out.items()}, first


def plot_equity_summary(r, folder):
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(1, r['equity'].shape[1] + 1)

    for curves, color, label in (
        (r['opt_equity'], viz.C_ORACLE, 'Oracle (sees the future)'),
        (r['equity'], viz.C_ACTOR, 'Actor'),
        (r['bh_equity'], viz.C_MARKET, 'Buy & hold'),
    ):
        lo, mid, hi = viz.iqr_bands(curves)
        viz.band(ax, x, lo, mid, hi, color, f'{label}  ({100 * (mid[-1] - 1):+.2f}% median)')

    ax.axhline(1.0, color=viz.REFERENCE, linewidth=1.2, linestyle='--')
    ax.annotate('break-even', xy=(x[-1], 1.0), xytext=(-4, -12), textcoords='offset points',
                ha='right', color=viz.INK_SOFT, fontsize=9)
    ax.set_title('Equity across test sequences (median, interquartile band)')
    ax.set_xlabel('Step (15m)')
    ax.set_ylabel('Equity')
    ax.legend(loc='upper left')
    viz.save(fig, f'{folder}/equity_summary.png')


def plot_outcomes(r, folder):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    lo = min(r['end_equity'].min(), r['bh_end_equity'].min())
    hi = max(r['end_equity'].max(), r['bh_end_equity'].max())
    bins = np.linspace(lo, hi, 60)
    ax1.hist(r['bh_end_equity'], bins=bins, color=viz.C_MARKET, alpha=0.55,
             label='Buy & hold', edgecolor=viz.SURFACE, linewidth=0.6)
    ax1.hist(r['end_equity'], bins=bins, histtype='step', color=viz.C_ACTOR,
             linewidth=2, label='Actor')
    ax1.axvline(1.0, color=viz.REFERENCE, linestyle='--', linewidth=1.2)
    ax1.set_title('End equity per sequence')
    ax1.set_xlabel('Equity')
    ax1.set_ylabel('Sequences')
    ax1.legend()

    dd_actor = 100 * max_drawdown(r['equity'])
    dd_bh = 100 * max_drawdown(r['bh_equity'])
    bins = np.linspace(0, max(dd_actor.max(), dd_bh.max()), 60)
    ax2.hist(dd_bh, bins=bins, color=viz.C_MARKET, alpha=0.55, label='Buy & hold',
             edgecolor=viz.SURFACE, linewidth=0.6)
    ax2.hist(dd_actor, bins=bins, histtype='step', color=viz.C_ACTOR, linewidth=2,
             label='Actor')
    ax2.set_title('Maximum drawdown per sequence')
    ax2.set_xlabel('Drawdown (%)')
    ax2.set_ylabel('Sequences')
    ax2.legend()

    viz.save(fig, f'{folder}/outcomes.png')


def plot_allocations(r, folder):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    actor = r['action'].ravel()
    oracle = r['opt_action'].ravel()
    bins = np.linspace(-1, 1, 61)
    ax1.hist(oracle, bins=bins, density=True, color=viz.C_ORACLE, alpha=0.55,
             label='Oracle', edgecolor=viz.SURFACE, linewidth=0.6)
    ax1.hist(actor, bins=bins, density=True, histtype='step', color=viz.C_ACTOR,
             linewidth=2, label='Actor')
    ax1.axvline(0.0, color=viz.REFERENCE, linestyle='--', linewidth=1.2)
    ax1.set_title('Allocation distribution')
    ax1.set_xlabel('Allocation')
    ax1.set_ylabel('Density')
    ax1.legend()

    H, _, _ = np.histogram2d(oracle, actor, bins=[48, 48],
                               range=[[-1, 1], [-1, 1]])
    viz.heatmap(ax2, (H / max(H.sum(), 1)).T, extent=[-1, 1, -1, 1],
                label='Fraction of steps')
    ax2.plot([-1, 1], [-1, 1], color=viz.REFERENCE, linestyle='--', linewidth=1.2)
    corr = np.corrcoef(oracle, actor)[0, 1]
    ax2.set_title(f'Actor vs oracle allocation (r = {corr:.3f})')
    ax2.set_xlabel('Oracle allocation')
    ax2.set_ylabel('Actor allocation')

    viz.save(fig, f'{folder}/allocations.png')


def plot_per_symbol(rows, folder):
    if not rows:
        return
    names = [r['symbol'] for r in rows]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(names) + 2.2))

    ax.barh(y - 0.2, [100 * r['roi_mean'] for r in rows], height=0.38,
            color=viz.C_ACTOR, label='Actor')
    ax.barh(y + 0.2, [100 * r['bh_roi'] for r in rows], height=0.38,
            color=viz.C_MARKET, label='Buy & hold')

    ax.axvline(0.0, color=viz.REFERENCE, linewidth=1.2)
    ax.set_yticks(y, names)
    ax.set_title('Mean ROI per sequence, by symbol')
    ax.set_xlabel('ROI (%)')
    ax.legend(loc='lower right')
    ax.grid(axis='y', visible=False)
    viz.save(fig, f'{folder}/per_symbol.png')


def plot_sequence(b, i, cfg, path):
    lo, hi = cfg['data']['returns']['min_value'], cfg['data']['returns']['max_value']
    horizon = b['action'].shape[1]
    t = np.arange(1, horizon + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10.5), sharex=True)

    ax1.axhline(1.0, color=viz.REFERENCE, linewidth=1.2, linestyle='--')
    ax1.plot(t, b['opt_equity'][i], color=viz.C_ORACLE,
             label=f"Oracle ({100 * (b['opt_end_equity'][i] - 1):+.2f}%)")
    ax1.plot(t, b['equity'][i], color=viz.C_ACTOR,
             label=f"Actor ({100 * (b['end_equity'][i] - 1):+.2f}%)")
    ax1.plot(t, b['bh_equity'][i], color=viz.C_MARKET, linestyle='--', linewidth=1.6,
             label=f"Buy & hold ({100 * (b['bh_end_equity'][i] - 1):+.2f}%)")
    ax1.set_title('Equity')
    ax1.set_ylabel('Equity')
    ax1.legend(loc='upper left')

    viz.heatmap(ax2, b['return_prob'][i].T, extent=[0.5, horizon + 0.5, lo, hi],
                label='Probability')
    ax2.scatter(t, b['return'][i].squeeze(-1), s=32, color=viz.C_MARKET,
                edgecolor=viz.SURFACE, linewidth=1.0, label='Realised', zorder=5)
    ax2.set_title('World-model return forecast')
    ax2.set_ylabel('Return (sigma)')
    ax2.legend(loc='upper left')

    viz.heatmap(ax3, b['action_prob'][i].T, extent=[0.5, horizon + 0.5, -1, 1],
                label='Probability')
    ax3.scatter(t, b['opt_action'][i].squeeze(-1), s=32, color=viz.C_ORACLE,
                edgecolor=viz.SURFACE, linewidth=1.0, label='Oracle', zorder=5)
    ax3.scatter(t, b['action'][i].squeeze(-1), s=38, marker='x', color=viz.C_ACTOR,
                linewidth=1.8, label='Actor', zorder=6)
    ax3.set_title('Allocation')
    ax3.set_xlabel('Step (15m)')
    ax3.set_ylabel('Allocation')
    ax3.legend(loc='upper left')

    viz.save(fig, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ar', action='store_true', help='Evaluate the autoregressive fine-tune.')
    parser.add_argument('--ckpt', default='best', choices=['best', 'last'])
    parser.add_argument('--decode', default='expected', choices=['expected', 'argmax', 'sample'])
    args = parser.parse_args()

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.to_container(OmegaConf.load('./config.yaml'), resolve=True)
    logger = set_logger(cfg)
    viz.use_style()

    folder = './figs/backtest'
    os.makedirs(folder, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    jepa = JEPA.load_from_checkpoint(f"./models/{cfg['jepa']['name']}/best.ckpt", cfg=cfg)

    cls = ActorAR if args.ar else Actor
    name = cfg['actor']['name'] + ('-AR' if args.ar else '')
    model = cls.load_from_checkpoint(f'./models/{name}/{args.ckpt}.ckpt', cfg=cfg, jepa=jepa)
    model.eval().to(device)

    dataset = CryptoDataset(cfg, mode='test', make_action=True)
    loader = DataLoader(dataset, batch_size=cfg['actor']['test']['batch_size'], shuffle=False)

    r, first = run_backtest(model, loader, device,
                            args.decode, cfg['actor']['test']['act_temp'], logger)

    commission = cfg['data']['actions']['commission_value']
    rows = [
        summarize(r['equity'], r['action'].squeeze(-1), commission, 'Actor'),
        summarize(r['bh_equity'], np.ones_like(r['action'].squeeze(-1)), commission, 'Buy & hold'),
        summarize(r['opt_equity'], r['opt_action'].squeeze(-1), commission, 'Oracle'),
    ]

    cols = ['label', 'roi_mean', 'roi_median', 'sharpe_ann', 'sortino_ann',
            'max_dd_mean', 'win_rate', 'profit_factor', 'turnover']
    heads = ['Strategy', 'ROI mean', 'ROI med', 'Sharpe(a)', 'Sortino(a)',
             'MaxDD', 'Win rate', 'Prof.fac', 'Turnover']
    logger.info('\n' + format_table(rows, cols, heads))

    actor = rows[0]
    logger.info('')
    logger.info(f"Actor mean ROI {100 * actor['roi_mean']:+.4f}% per 16h sequence, "
                f"95% CI [{100 * actor['roi_ci_lo']:+.4f}%, {100 * actor['roi_ci_hi']:+.4f}%], "
                f"t = {actor['t_stat']:.2f}")
    logger.info(f"Exposure: {100 * actor['frac_long']:.1f}% long / "
                f"{100 * actor['frac_short']:.1f}% short / {100 * actor['frac_flat']:.1f}% flat; "
                f"mean |allocation| {actor['exposure_abs']:.3f}")
    logger.info(f"Commission drag over a sequence: {100 * actor['cost_drag']:.4f}%")

    # Per symbol. A pooled number can hide a policy that only works on one market.
    per_symbol = []
    for sid, symbol in enumerate(dataset.symbol_names):
        m = r['symbol_id'] == sid
        if m.sum() < 2:
            continue
        s = summarize(r['equity'][m], r['action'][m].squeeze(-1), commission, symbol)
        s['symbol'] = symbol
        s['bh_roi'] = float(np.mean(r['bh_end_equity'][m]) - 1.0)
        s['excess'] = s['roi_mean'] - s['bh_roi']
        per_symbol.append(s)

    if per_symbol:
        logger.info('\nPer symbol:\n' + format_table(
            per_symbol,
            ['symbol', 'sequences', 'roi_mean', 'bh_roi', 'excess', 'sortino_ann', 'win_rate'],
            ['Symbol', 'Seqs', 'ROI', 'B&H ROI', 'Excess', 'Sortino(a)', 'Win rate']))

    plot_equity_summary(r, folder)
    plot_outcomes(r, folder)
    plot_allocations(r, folder)
    plot_per_symbol(per_symbol, folder)
    for i in range(min(NUM_PLOT_SEQS, first['action'].shape[0])):
        plot_sequence(first, i, cfg, f'{folder}/backtest{i + 1}.png')

    payload = {
        'decode': args.decode,
        'checkpoint': f'{name}/{args.ckpt}',
        'steps_per_year': STEPS_PER_YEAR,
        'overall': rows,
        'per_symbol': per_symbol,
    }
    with open(f'{folder}/metrics.json', 'w') as f:
        json.dump(payload, f, indent=2, default=float)

    logger.info(f"\nFigures and metrics.json written to '{folder}'")


if __name__ == '__main__':
    main()
