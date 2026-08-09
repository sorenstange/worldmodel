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
from data import CryptoDataset
from util import *

def model_data_wrapper(model, batch):
    if torch.cuda.is_available():
        model.cuda()
        X = batch['sample'].cuda()      
        t_ret = batch['return'].cuda() 
        t_ret_idx = batch['return_target'].cuda() 
        t_act = batch['action'].cuda()
        t_act_idx = batch['action_target'].cuda()
    else:
        model.cpu()
        X = batch['sample'].cpu()
        t_ret = batch['return'].cpu()
        t_ret_idx = batch['return_target'].cpu() 
        t_act = batch['action'].cpu()
        t_act_idx = batch['action_target'].cpu()

    return model, {
        'raw_states' : X,
        't_ret' : t_ret,
        't_ret_idx' : t_ret_idx,
        't_act' : t_act,
        't_act_idx' : t_act_idx
    }


def dream_output_wrapper(dream_output):
    d_output = []
    for d_item in dream_output:
        d_output.append(d_item.detach().cpu().numpy())

    (d_states, d_ret, d_ret_probs, d_ret_sampled_idx,
    d_act, d_act_probs, d_act_sampled_idx) = d_output

    return {
        'd_states' : d_states,
        'd_ret' : d_ret,
        'd_ret_probs' : d_ret_probs,
        'd_ret_idx' : d_ret_sampled_idx,
        'd_act' : d_act,
        'd_act_probs' : d_act_probs,
        'd_act_idx' : d_act_sampled_idx
    }

def future_data_wrapper(batch, Z, start_idx):
    t_states = Z[:, start_idx+1:, :].detach().cpu().numpy()
    t_ret = batch['t_ret'][:, start_idx+1:, :].cpu().numpy()
    t_ret_idx = batch['t_ret_idx'][:, start_idx+1:, :].cpu().numpy()
    t_act = batch['t_act'][:, start_idx+1:, :].cpu().numpy()
    t_act_idx = batch['t_act_idx'][:, start_idx+1:, :].cpu().numpy()

    return {
        't_states' : t_states,
        't_ret' : t_ret,
        't_ret_idx' : t_ret_idx,
        't_act' : t_act,
        't_act_idx' : t_act_idx
    }

def prepare_dplot_dict(dplot_dict, return_bins, action_bins, commission_value):
    dplot_dict['t_cumprod'] = np.cumprod(1 + dplot_dict['t_ret'].squeeze(), axis = 1)
    dplot_dict['d_cumprod'] = np.cumprod(1 + dplot_dict['d_ret'].squeeze(), axis = 1)

    dplot_dict['t_ret_vals'] = return_bins[dplot_dict['t_ret_idx']].detach().cpu().numpy().squeeze()
    dplot_dict['d_ret_vals'] = return_bins[dplot_dict['d_ret_idx']].detach().cpu().numpy().squeeze()

    t_a = torch.tensor(dplot_dict['t_act'])
    d_a = torch.tensor(dplot_dict['d_act'])
    act0 = torch.full((t_a.size(0), 1, 1), 0.0, dtype=t_a.dtype, device=t_a.device)

    t_a = torch.cat((act0, t_a), dim=1).squeeze(-1)
    d_a = torch.cat((act0, d_a), dim=1).squeeze(-1)

    t_r = torch.tensor(dplot_dict['t_ret']).squeeze(-1)
    d_r = torch.tensor(dplot_dict['d_ret']).squeeze(-1)

    dplot_dict['t_equity'], _ = equity(t_a, t_r, commission_value)
    dplot_dict['d_equity'], _ = equity(d_a, d_r, commission_value)

    dplot_dict['t_equity'] = dplot_dict['t_equity'].detach().cpu().squeeze(0).numpy()
    dplot_dict['d_equity'] = dplot_dict['d_equity'].detach().cpu().squeeze(0).numpy()

    dplot_dict['t_act_vals'] = action_bins[dplot_dict['t_act_idx']].detach().cpu().numpy()
    dplot_dict['d_act_vals'] = action_bins[dplot_dict['d_act_idx']].detach().cpu().numpy()

    return dplot_dict

def plot_dream(t_steps, dplot_dict, cfg, i, figure_name):
    min_return = cfg['data']['returns']['min_value']
    max_return = cfg['data']['returns']['max_value']

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # ax1 subplot
    ax1.plot(t_steps, dplot_dict['t_cumprod'][i, :], label='True Cumprod', color='black', linewidth=2.5)
    ax1.plot(t_steps, dplot_dict['d_cumprod'][i, :], label=f'Dream Cumprod', color='darkorange', linestyle=':', linewidth=2)
    ax1.set_title(f'JEPA Worldmodel test', fontsize=14)
    ax1.set_ylabel('Cumulative return', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # ax2 subplot
    im1 = ax2.imshow(dplot_dict['d_ret_probs'][i, :].T, aspect='auto', cmap='viridis', origin='lower',
                    extent=[0.5, horizon + 0.5, min_return, max_return])

    ax2.scatter(t_steps, dplot_dict['t_ret_vals'][i, :], color='red', edgecolor='white', s=45, label='True return', zorder=5)
    ax2.scatter(t_steps, dplot_dict['d_ret_vals'][i, :], color='cyan', marker='x', s=55, linewidths=2, label='Dream return', zorder=6)
    ax2.set_title(f'Probability Heatmap', fontsize=14)
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Returns', fontsize=12)
    ax2.set_xticks(t_steps)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2)) 
    ax2.legend(fontsize=11, loc='upper left')
    cbar = fig.colorbar(im1, ax=ax2, orientation='vertical', pad=0.005)
    cbar.set_label('Return probability', fontsize=11)

    # ax3 subplot
    ax3.plot(t_steps, dplot_dict['t_equity'][i, :], label='True strategy', color='black', linewidth=2.5)
    ax3.plot(t_steps, dplot_dict['d_equity'][i, :], label=f'Dream strategy', color='darkorange', linestyle=':', linewidth=2)
    ax3.set_title(f'Strategy test', fontsize=14)
    ax3.set_ylabel('Cumulative return', fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # ax4 subplot
    im2 = ax4.imshow(dplot_dict['d_act_probs'][i, :].T, aspect='auto', cmap='viridis', origin='lower',
                    extent=[0.5, horizon + 0.5, -1.0, 1.0])
    ax4.scatter(t_steps, dplot_dict['t_act_vals'][i, :], color='red', edgecolor='white', s=45, label='True action', zorder=5)
    ax4.scatter(t_steps, dplot_dict['d_act_vals'][i, :], color='cyan', marker='x', s=55, linewidths=2, label='Dream action', zorder=6)
    ax4.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2)) 
    ax4.legend(fontsize=11, loc='upper left')
    cbar = fig.colorbar(im2, ax=ax4, orientation='vertical', pad=0.005)
    cbar.set_label('Action probability', fontsize=11)

    plt.tight_layout()

    plt.savefig(figure_name, dpi=300)
    plt.close()

def backtest_output_wrapper(backtest_output):
    b_output = []
    for b_item in backtest_output:
        b_output.append(b_item.detach().cpu().numpy())

    (b_state, b_ret_probs, b_act, b_act_probs, b_act_idx) = b_output

    return {
        'b_state' : b_state,
        'b_ret_probs' : b_ret_probs,
        'b_act' : b_act,
        'b_act_probs' : b_act_probs,
        'b_act_idx' : b_act_idx
    }

def prepare_bplot_dict(bplot_dict, return_bins, action_bins, commission_value):
    bplot_dict['t_cumprod'] = np.cumprod(1 + bplot_dict['t_ret'], axis = 1)
    bplot_dict['t_ret_vals'] = return_bins[bplot_dict['t_ret_idx']].detach().cpu().numpy()

    t_a = torch.tensor(bplot_dict['t_act'])
    b_a = torch.tensor(bplot_dict['b_act'])
    act0 = torch.full((t_a.size(0), 1, 1), 0.0, dtype=t_a.dtype, device=t_a.device)

    t_a = torch.cat((act0, t_a), dim=1).squeeze(-1)
    b_a = torch.cat((act0, b_a), dim=1).squeeze(-1)

    t_r = torch.tensor(bplot_dict['t_ret']).squeeze(-1)

    bplot_dict['t_equity'], _ = equity(t_a, t_r, commission_value)
    bplot_dict['b_equity'], _ = equity(b_a, t_r, commission_value)

    bplot_dict['t_equity'] = bplot_dict['t_equity'].detach().cpu().squeeze(0).numpy()
    bplot_dict['b_equity'] = bplot_dict['b_equity'].detach().cpu().squeeze(0).numpy()

    bplot_dict['t_act_vals'] = action_bins[bplot_dict['t_act_idx']].detach().cpu().numpy()
    bplot_dict['b_act_vals'] = action_bins[bplot_dict['b_act_idx']].detach().cpu().numpy()

    return bplot_dict

def plot_backtest(t_steps, bplot_dict, cfg, i, figure_name):
    min_return = cfg['data']['returns']['min_value']
    max_return = cfg['data']['returns']['max_value']

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # ax1 subplot
    ax1.plot(t_steps, bplot_dict['t_cumprod'][i, :], label='True Cumprod', color='black', linewidth=2.5)
    ax1.set_title(f'JEPA Worldmodel test', fontsize=14)
    ax1.set_ylabel('Cumulative return', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # ax2 subplot
    im1 = ax2.imshow(bplot_dict['b_ret_probs'][i, :].T, aspect='auto', cmap='viridis', origin='lower',
                    extent=[0.5, horizon + 0.5, min_return, max_return])
    ax2.scatter(t_steps, bplot_dict['t_ret_vals'][i, :], color='red', edgecolor='white', s=45, label='True return', zorder=5)
    ax2.set_title(f'Probability Heatmap', fontsize=14)
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Returns', fontsize=12)
    ax2.set_xticks(t_steps)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2)) 
    ax2.legend(fontsize=11, loc='upper left')
    cbar = fig.colorbar(im1, ax=ax2, orientation='vertical', pad=0.005)
    cbar.set_label('Return probability', fontsize=11)

    # ax3 subplot
    ax3.plot(t_steps, bplot_dict['t_equity'][i, :], label='True strategy', color='black', linewidth=2.5)
    ax3.plot(t_steps, bplot_dict['b_equity'][i, :], label=f'Dream strategy', color='darkorange', linestyle=':', linewidth=2)
    ax3.set_title(f'Strategy test', fontsize=14)
    ax3.set_ylabel('Cumulative return', fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # ax4 subplot
    im2 = ax4.imshow(bplot_dict['b_act_probs'][i, :].T, aspect='auto', cmap='viridis', origin='lower',
                    extent=[0.5, horizon + 0.5, -1.0, 1.0])
    ax4.scatter(t_steps, bplot_dict['t_act_vals'][i, :], color='red', edgecolor='white', s=45, label='True action', zorder=5)
    ax4.scatter(t_steps, bplot_dict['b_act_vals'][i, :], color='cyan', marker='x', s=55, linewidths=2, label='Dream action', zorder=6)
    ax4.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2)) 
    ax4.legend(fontsize=11, loc='upper left')
    cbar = fig.colorbar(im2, ax=ax4, orientation='vertical', pad=0.005)
    cbar.set_label('Action probability', fontsize=11)

    plt.tight_layout()

    plt.savefig(figure_name, dpi=300)
    plt.close()

if __name__ == '__main__':
    ####### CONTROL PARAMETERS #######
    batch_size = 32
    horizon = 32
    ret_temperature = 1.0 
    act_temperature = 1.0
    num_dream_plots = 20; D = 0
    num_backtest_plots = 20; B = 0
    ##################################

    # 0. INITIALIZATION
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.load('./config.yaml')
    cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = set_logger(cfg)    

    model = JEPA.load_from_checkpoint(f'./models/jepa/{cfg['jepa']['name']}/last.ckpt', cfg=cfg, weights_only=False)
    model.eval()

    test_dataset = CryptoDataset(cfg, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(test_loader))
    
    model, batch = model_data_wrapper(model, batch)

    with torch.no_grad():
        Z_true = model.encode(batch['raw_states'])

    start_idx = Z_true.size(1) - horizon - 1
    t_steps = np.arange(1, horizon + 1)
    f_dict = future_data_wrapper(batch, Z_true, start_idx)

    # 1. START OF DREAM PLOTTING
    folder = './figs/dreams'
    plot_name = '/dream'
    os.makedirs(folder, exist_ok = True)
    loss_fn = make_constrained_loss(loss_fn_so)    

    Z_prompt = Z_true[:, :start_idx+1, :]
    Ret_prompt = batch['t_ret'][:, :start_idx+1, :]
    Act_prompt = optimal_allocation(Ret_prompt.squeeze(-1), c = 0.0005, loss_fn = loss_fn) 
    Act_prompt = Act_prompt[:, 1:].unsqueeze(-1)

    with torch.no_grad():
        dream_output = model.dream( Z_prompt, Ret_prompt, Act_prompt, 
                                    horizon = horizon, 
                                    ret_temperature = ret_temperature,
                                    act_temperature = act_temperature
                                    )

    d_dict = dream_output_wrapper(dream_output)
    dplot_dict = d_dict | f_dict
    dplot_dict = prepare_dplot_dict(dplot_dict, model.return_bins, model.action_bins, cfg['data']['actions']['commission_value'])

    while D < num_dream_plots:
        figure_name = folder + plot_name + f'{D+1}.png'
        MSE = np.mean((dplot_dict['t_states'][D, :] - dplot_dict['d_states'][D, :]) ** 2)
        plot_dream(t_steps, dplot_dict, cfg, D, figure_name)
        logger.info(f'Run {D+1}: MSE in the latent space over {horizon} steps: {MSE:.4f}')
        D += 1

    logger.info(f"Done! Dream figures saved in '{folder}'")

    # 2. START OF BACKTEST PLOTTING
    folder = './figs/backtest'
    plot_name = '/backtest'
    os.makedirs(folder, exist_ok = True)
    
    end_equity = []

    for batch in test_loader:
        model, batch = model_data_wrapper(model, batch)
        with torch.no_grad():
            Z_true = model.encode(batch['raw_states'])

        start_idx = Z_true.size(1) - horizon - 1
        t_steps = np.arange(1, horizon + 1)
        f_dict = future_data_wrapper(batch, Z_true, start_idx)

        with torch.no_grad():
            backtest_output = model.backtest(Z_true, batch['t_ret'], horizon = horizon, act_temperature = 1.0)

        b_dict = backtest_output_wrapper(backtest_output)
        bplot_dict = b_dict | f_dict
        bplot_dict = prepare_bplot_dict(bplot_dict, model.return_bins, model.action_bins, cfg['data']['actions']['commission_value'])
        end_equity.append(bplot_dict['b_equity'][:, -1])

        while B < num_backtest_plots:
            figure_name = folder + plot_name + f'{B+1}.png'
            plot_backtest(t_steps, bplot_dict, cfg, B, figure_name)
            logger.info(f'Run {B+1}: End equity: {bplot_dict['b_equity'][B,-1]:.4f}')
            B += 1

    end_equity = np.concatenate(end_equity)
    logger.info(f'Avg. End equity: {np.mean(end_equity):.4f}')
    logger.info(f"Done! Backtest figures saved in '{folder}'")
