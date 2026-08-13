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
    else:
        model.cpu()
        X = batch['sample'].cpu()
        t_ret = batch['return'].cpu()
        t_ret_idx = batch['return_target'].cpu() 

    return model, {
        'raw_states' : X,
        't_ret' : t_ret,
        't_ret_idx' : t_ret_idx
    }

def dream_output_wrapper(dream_output):
    d_output = []
    for d_item in dream_output:
        d_output.append(d_item.detach().cpu().numpy())

    (d_states, d_ret, d_ret_probs, d_ret_sampled_idx) = d_output

    return {
        'd_states' : d_states,
        'd_ret' : d_ret,
        'd_ret_probs' : d_ret_probs,
        'd_ret_idx' : d_ret_sampled_idx
    }

def future_data_wrapper(batch, Z, start_idx):
    t_states = Z[:, start_idx+1:, :].detach().cpu().numpy()
    t_ret = batch['t_ret'][:, start_idx+1:, :].cpu().numpy()
    t_ret_idx = batch['t_ret_idx'][:, start_idx+1:, :].cpu().numpy()

    return {
        't_states' : t_states,
        't_ret' : t_ret,
        't_ret_idx' : t_ret_idx
    }

def prepare_dplot_dict(dplot_dict, return_bins):
    dplot_dict['t_cumprod'] = np.cumprod(1 + dplot_dict['t_ret'].squeeze(), axis = 1)
    dplot_dict['d_cumprod'] = np.cumprod(1 + dplot_dict['d_ret'].squeeze(), axis = 1)

    dplot_dict['t_ret_vals'] = return_bins[dplot_dict['t_ret_idx']].detach().cpu().numpy().squeeze()
    dplot_dict['d_ret_vals'] = return_bins[dplot_dict['d_ret_idx']].detach().cpu().numpy().squeeze()

    return dplot_dict

def plot_dream(t_steps, dplot_dict, cfg, i, figure_name):
    min_return = cfg['data']['returns']['min_value']
    max_return = cfg['data']['returns']['max_value']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # ax1 subplot
    ax1.plot(t_steps, dplot_dict['t_cumprod'][i, :], label='True Cumprod', color='black', linewidth=2.5)
    ax1.plot(t_steps, dplot_dict['d_cumprod'][i, :], label=f'Dream Cumprod', color='darkorange', linestyle=':', linewidth=2)
    ax1.set_title(f'JEPA Worldmodel test', fontsize=14)
    ax1.set_ylabel('Cumulative return', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # ax2 subplot
    im1 = ax2.imshow(dplot_dict['d_ret_probs'][i, :].T, aspect='auto', cmap='viridis', origin='lower',
                    extent=[0.5, horizon + 0.5, min_return, max_return], vmin=0.0, vmax=1.0)

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

    plt.tight_layout()

    plt.savefig(figure_name, dpi=300)
    plt.close()

if __name__ == '__main__':
    ####### CONTROL PARAMETERS #######
    batch_size = 32
    horizon = 32
    ret_temperature = 1.0
    num_dream_plots = 20; D = 0
    ##################################

    # 0. INITIALIZATION
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.load('./config.yaml')
    cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = set_logger(cfg)    
    logger.info('Starting JEPA-test pipeline')

    model = JEPA.load_from_checkpoint(f'./models/{cfg['jepa']['name']}/last.ckpt')
    model.eval()

    test_dataset = CryptoDataset(cfg, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    folder = './figs/dreams'
    plot_name = '/dream'
    os.makedirs(folder, exist_ok = True)
    
    batch = next(iter(test_loader))

    model, batch = model_data_wrapper(model, batch)

    with torch.no_grad():
        Z_true = model.encode(batch['raw_states'])

    start_idx = Z_true.size(1) - horizon - 1
    t_steps = np.arange(1, horizon + 1)
    f_dict = future_data_wrapper(batch, Z_true, start_idx)

    Z_prompt = Z_true[:, :start_idx+1, :]
    Ret_prompt = batch['t_ret'][:, :start_idx+1, :]

    with torch.no_grad():
        dream_output = model.dream(Z_prompt, Ret_prompt, horizon = horizon, ret_temperature = ret_temperature)

    d_dict = dream_output_wrapper(dream_output)
    dplot_dict = d_dict | f_dict
    dplot_dict = prepare_dplot_dict(dplot_dict, model.return_bins)

    while D < num_dream_plots:
        figure_name = folder + plot_name + f'{D+1}.png'
        MSE = np.mean((dplot_dict['t_states'][D, :] - dplot_dict['d_states'][D, :]) ** 2)
        plot_dream(t_steps, dplot_dict, cfg, D, figure_name)
        logger.info(f'Run {D+1}: MSE in the latent space over {horizon} steps: {MSE:.4f}')
        D += 1

    logger.info(f"Done! Dream figures saved in '{folder}'")
