import torch
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import logging
import os

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from jepa import JEPA
from data import CryptoDataset
from util import *

if __name__ == '__main__':
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.load('./config.yaml')
    cfg = OmegaConf.to_container(cfg, resolve=True)

    logger = set_logger(cfg)

    batch_size = 1
    horizon = 30
    ret_temperature = 1.0 
    act_temperature = 1.0

    test_dataset = CryptoDataset(cfg, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    batch = next(iter(test_loader))
    
    model = JEPA.load_from_checkpoint(f'./models/jepa/{cfg['jepa']['name']}/last.ckpt', cfg=cfg, weights_only=False)
    model.eval()
    if torch.cuda.is_availabe():
        model.cuda()
        X = batch['sample'].cuda()      
        ret_true = batch['return'].cuda() 
        ret_true_target = batch['return_target'].cuda() 
        act_true = batch['action'].cuda()
        act_true_target = batch['action_target'].cuda()
    else:
        X = batch['sample']   
        ret_true = batch['return'] 
        ret_true_target = batch['return_target']
        act_true = batch['action']
        act_true_target = batch['action_target']

    with torch.no_grad():
        Z_true = model.encode(X)
        start_idx = Z_true.size(1) - horizon - 1
        
        Z_prompt = Z_true[:, :start_idx+1, :]          # [1, T_hist, d_model]
        Ret_prompt = ret_true[:, :start_idx+1, :]      # [1, T_hist, 1]
        Act_prompt = act_true[:, :start_idx+1, :]

        (dream_states,
        dream_ret,
        dream_ret_probs,
        dream_ret_sampled_idx,
        dream_act,
        dream_act_probs,
        dream_act_sampled_idx) = model.imagine(Z_hist, Ret_hist, Act_prompt, 
                                                horizon = horizon, 
                                                ret_temperature = ret_temperature,
                                                act_temperature = act_temperature)
    Z_pred = dream_states
    Z_true = Z_true[:, start_idx+1:, :]

    MSE = np.mean((Z_true - Z_pred) ** 2)
    logger.info(f'MSE in the latent space over {horizon} steps: {MSE:.4f}')

    true_ret = ret_true[:, :start_idx+1, :].cpu().numpy()
    dream_ret = dream_ret.detach().cpu().numpy()

    true_cumprod = np.cumprod(1 + true_ret)
    dream_cumprod = np.cumprod(1 + dream_ret)

        # ==================== PLOTTING DELEN ====================
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    t_steps = np.arange(1, horizon + 1)

    # --- ØVERSTE SUBPLOT: Kumulativt Afkast ---
    ax1.plot(t_steps, true_cumprod, label='True Cumprod', color='black', linewidth=2.5)
    ax1.plot(t_steps, dream_cumprod, label=f'Dream Cumprod (Temp={ret_temperature})', color='darkorange', linestyle=':', linewidth=2)
    ax1.set_title(f'JEPA Worldmodel test', fontsize=14)
    ax1.set_ylabel('Cumulative return (1.0 = Start)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- NEDERSTE SUBPLOT: Heatmap over sandsynligheder ---
    # Hent min og max værdier direkte fra dine bin_centers
    min_return = cfg['data']['returns']['min_value']
    max_return = cfg['data']['returns']['max_value']

    # FIX 1: Vi ændrer 'extent' i bunden og toppen til de reelle return-værdier (floats)
    im = ax2.imshow(dream_ret_probs.detach().cpu().numpy().T, aspect='auto', cmap='viridis', origin='lower',
                    extent=[0.5, horizon + 0.5, min_return, max_return])
    
    # FIX 2: Da y-aksen nu er numeriske returns, skal vi mappe y-værdierne til bin_centers i stedet for indekser
    true_returns_vals = model.return_bins[ret_true_target[:, :start_idx+1, :].cpu().numpy()]
    sampled_returns_vals = model.return_bins[dream_ret_sampled_idx.detach().cpu().numpy()]

    # Plot punkterne ved de korrekte float-værdier på y-aksen
    ax2.scatter(t_steps, true_returns_vals, color='red', edgecolor='white', s=45, label='True return', zorder=5)
    ax2.scatter(t_steps, sampled_returns_vals, color='cyan', marker='x', s=55, linewidths=2, label='Dream return', zorder=6)

    ax2.set_title(f'Conditional Probability Heatmap (Temp={ret_temperature})', fontsize=14)
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Returns', fontsize=12)
    ax2.set_xticks(t_steps)

    # FIX 3:
    act0 = torch.full((batch_size, 1), x0, dtype=p.dtype, device=p.device)
    true_act = act_true[:, start_idx+1:, :]

    dream_equity = equity(torch.cat((act0, dream_act), dim=-1), dream_ret, cfg['data']['actions']['commission_value'])
    true_equity = equity(torch.cat((act0, true_act), dim=-1), true_ret, cfg['data']['actions']['commission_value'])

    ax3.plot(t_steps, true_equity, label='True optimized strategy', color='black', linewidth=2.5)
    ax3.plot(t_steps, dream_equity, label=f'Dream strategy (Temp={act_temperature})', color='darkorange', linestyle=':', linewidth=2)
    ax3.set_title(f'Strategy test', fontsize=14)
    ax3.set_ylabel('Cumulative return (1.0 = Start)', fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # FIX 4:
    im2 = ax4.imshow(dream_act_probs.detach().cpu().numpy().T, aspect='auto', cmap='viridis', origin='lower',
                    extent=[0.5, horizon + 0.5, -1.0, 1.0])

    true_returns_vals = model.action_bins[act_true_target[:, :start_idx+1, :].cpu().numpy()]
    sampled_returns_vals = model.action_bins[dream_act_sampled_idx.detach().cpu().numpy()]
    
    # FIX 3: Formater y-aksens labels pænt som procenter
    import matplotlib.ticker as mtick
    # Viser værdier med procenttegn og 2 decimaler (f.eks. 0.05 bliver til 5.00%)
    # Da dine returns er rå procenter (f.eks. 0.001 for 0.1%), ganger vi ticks med 100
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2)) 
    
    ax2.legend(fontsize=11, loc='upper left')

    cbar = fig.colorbar(im, ax=ax2, orientation='vertical', pad=0.02)
    cbar.set_label('Temperature-scale probability', fontsize=11)

    plt.tight_layout()
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/jepa_trajectory_diagnostic.png", dpi=300)
    logger.info("Done! Figure saved in 'figs/jepa_trajectory_diagnostic.png'")
