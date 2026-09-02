import torch
import lightning as L
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import logging
import os

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from jepa import JEPA
from actor import Actor
from data import CryptoDataset
from util import *

def plot_backtest(backtest_output, cfg, B, figure_name):
    min_return = cfg['data']['returns']['min_value']
    max_return = cfg['data']['returns']['max_value']
    horizon = backtest_output['action'].shape[1]
    t_steps = np.arange(1, horizon + 1)

    t_cumprod = np.cumprod(1 + backtest_output['return'][B, :], axis = 1)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # ax1 subplot
    ax1.plot(t_steps, t_cumprod, label='Market movement', color='black', linewidth=2.5)
    ax1.set_title(f'JEPA Actor test', fontsize=14)
    ax1.set_ylabel('Cumulative return', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # ax2 subplot
    im1 = ax2.imshow(backtest_output['return_prob'][B, :].T, aspect='auto', cmap='viridis', origin='lower',
                    extent=[0.5, horizon + 0.5, min_return, max_return], vmin=0.0, vmax=1.0)
    ax2.scatter(t_steps, backtest_output['return'][B, :], color='red', edgecolor='white', s=45, label='True return', zorder=5)
    ax2.set_title(f'Probability Heatmap', fontsize=14)
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Returns', fontsize=12)
    ax2.set_xticks(t_steps)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2)) 
    ax2.legend(fontsize=11, loc='upper left')
    cbar = fig.colorbar(im1, ax=ax2, orientation='vertical', pad=0.005)
    cbar.set_label('Return probability', fontsize=11)

    # ax3 subplot
    ax3.plot(t_steps, backtest_output['opt_equity'][B, :], label=f'Optimal strategy ({100*(backtest_output['opt_end_equity'][B]-1):.2f}%)', color='black', linewidth=2.5)
    ax3.plot(t_steps, backtest_output['equity'][B, :], label=f'Actor strategy  ({100*(backtest_output['end_equity'][B]-1):.2f}%)', color='darkorange', linestyle=':', linewidth=2)
    ax3.set_title(f'Strategy test', fontsize=14)
    ax3.set_ylabel('Cumulative return', fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # ax4 subplot
    im2 = ax4.imshow(backtest_output['action_prob'][B, :].T, aspect='auto', cmap='viridis', origin='lower',
                    extent=[0.5, horizon + 0.5, -1.0, 1.0], vmin=0.0, vmax=1.0)
    ax4.scatter(t_steps, backtest_output['opt_action'][B, :], color='red', edgecolor='white', s=45, label='True action', zorder=5)
    ax4.scatter(t_steps, backtest_output['action'][B, :], color='cyan', marker='x', s=55, linewidths=2, label='Actor action', zorder=6)
    ax4.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2)) 
    ax4.legend(fontsize=11, loc='upper left')
    cbar = fig.colorbar(im2, ax=ax4, orientation='vertical', pad=0.005)
    cbar.set_label('Action probability', fontsize=11)

    plt.tight_layout()

    plt.savefig(figure_name, dpi=300)
    plt.close()

if __name__ == '__main__':
    # 0. INITIALIZATION
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.load('./config.yaml')
    cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = set_logger(cfg)    
    logger.info('Starting Actor-test pipeline')

    jepa = JEPA.load_from_checkpoint(f'./models/{cfg['jepa']['name']}/best.ckpt', cfg=cfg)
    model = Actor.load_from_checkpoint(f'./models/{cfg['actor']['name']}/last.ckpt', cfg=cfg, jepa=jepa)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    test_dataset = CryptoDataset(cfg, mode='test', make_action=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['actor']['test']['batch_size'], shuffle=True)

    folder = './figs/backtest'
    plot_name = '/backtest'
    os.makedirs(folder, exist_ok = True)

    B = 0
    end_equity = []
    t_end_equity = []
    for batch in test_loader:
        if torch.cuda.is_available():
            for key, item in batch.items():
                batch[key] = item.cuda()

        backtest_output = model.backtest(batch, cfg['actor']['test']['act_temp'])
        
        for key, item in backtest_output.items():
            backtest_output[key] = item.detach().cpu().numpy()

        end_equity.append(backtest_output['end_equity'])
        t_end_equity.append(backtest_output['opt_end_equity'])

        while B < cfg['actor']['test']['num_plots']:
            figure_name = folder + plot_name + f'{B+1}.png'
            plot_backtest(backtest_output, cfg, B, figure_name)
            logger.info(f'Run {B+1}: End equity: {backtest_output['end_equity'][B]:.4f}')
            B += 1

    end_equity = np.concatenate(end_equity)
    t_end_equity = np.concatenate(t_end_equity)
    
    logger.info(f'Avg. ROI: {100*(np.mean(end_equity)-1):.2f}%')
    logger.info(f'Avg. (Optimal) ROI: {100*(np.mean(t_end_equity)-1):.2f}%')
    logger.info(f"Done! Backtest figures saved in '{folder}'")
        

