import torch
import lightning as L
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from jepa import JEPA
from util import *
from data import get_OHLCV
import pandas as pd
import time

def process_data(df, cfg):
    df['Return'] = df['Close'].pct_change()

    mu = df['Close'].rolling(window = cfg['normalization_window']).mean()
    std = df['Close'].rolling(window = cfg['normalization_window']).std()
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = (df[col] - mu) / (std + 1e-8)

    df['Volume'] = np.log1p(df['Volume'])
    df['Volume'] = (df['Volume'] - df['Volume'].rolling(window = cfg['normalization_window']).mean()) / (df['Volume'].rolling(window = cfg['normalization_window']).std() + 1e-8)
    
    df['Volatility'] = std / mu

    df.dropna(inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Volatility', 'Return']].copy()
    return df.drop('Return', axis = 1).values.astype(np.float32), df['Return'].values.astype(np.float32)

if __name__ == '__main__':
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.load('./config.yaml')
    cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = set_logger(cfg)    
    logger.info('Starting JEPA-test pipeline')

    model = JEPA.load_from_checkpoint(f'./models/jepa/{cfg['jepa']['name']}/last.ckpt')
    model.cuda()
    model.eval()

    SYMBOL = 'WLDUSDT'
    TIMEFRAME = '1m'
    logger.info(f'Downloading data of {SYMBOL}...')
    df = get_OHLCV(SYMBOL, TIMEFRAME, SINCE = '2026-08-01 00:00', TO = None)
    logger.info(f'Downloading complete!')
    X, Ret = process_data(df, cfg['data'])
    X = torch.from_numpy(X)
    Ret = torch.from_numpy(Ret)

    X = X.unfold(0, cfg['data']['window_size'], cfg['data']['window_size']).transpose(1, 2)
    Ret = (Ret + 1.).unfold(0, cfg['data']['window_size'], cfg['data']['window_size'])
    Ret = (Ret.cumprod(1) - 1.)[:, -1]

    Seq = X.size(0)

    X = X.unsqueeze(0)
    Ret = Ret.unsqueeze(0).unsqueeze(-1)

    X = X.cuda()
    Ret = Ret.cuda()

    loss_fn = make_constrained_loss(loss_fn_so)

    ctx_win = cfg['jepa']['predictor']['max_len']

    with torch.no_grad():
        Z = model.encode(X)
    
    Z_p = Z[:, :ctx_win+1, :]
    Ret_p = Ret[:, :ctx_win+1, :]
    # Act_p = optimal_allocation(Ret_p.squeeze(-1), c = cfg['data']['actions']['commission_value'], loss_fn = loss_fn) 
    # Act_p = Act_p[:, 1:].unsqueeze(-1)
    Act_p = torch.zeros_like(Ret_p, device = Ret_p.device)

    actions = []; ret = []
    Z_p = truncate(Z_p, ctx_win)
    Ret_p = truncate(Ret_p, ctx_win)
    Act_p = truncate(Act_p, ctx_win)

    for t in range(Seq-ctx_win-1):
        with torch.no_grad():
            Z_next, ret_logits, act_logits = model.predict(Z_p, Ret_p, Act_p)

        new_Z = Z[:, ctx_win+t+1, :].unsqueeze(1)
        new_ret = Ret[:, ctx_win+t+1, :].unsqueeze(1)

        act_logits = act_logits[:, -1:, :].squeeze(1)
        
        act_probs = torch.softmax(act_logits, dim=-1)
        act_sampled_idx = torch.multinomial(act_probs, num_samples=1)
        new_act = model.action_bins[act_sampled_idx].unsqueeze(-1)
        
        #act_idx = torch.argmax(act_logits)
        #new_act = model.action_bins[act_idx].reshape((1,1,-1))
        
        Z_prompt = torch.cat([Z_p, new_Z], dim=1)
        Ret_prompt = torch.cat([Ret_p, new_ret], dim=1)
        Act_p = torch.cat([Act_p, new_act], dim=1)

        Z_p = truncate(Z_p, ctx_win)
        Ret_p = truncate(Ret_p, ctx_win)
        Act_p = truncate(Act_p, ctx_win)
        actions.append(new_act)
    
    actions = torch.concat(actions).reshape(1, -1).unsqueeze(-1)
    #print('actions', actions)
    act0 = torch.full((1, 1, 1), 0.0, dtype=actions.dtype, device=actions.device)
    t_a = torch.cat((act0, actions), dim=1).squeeze(-1)
    Ret_f = Ret[:, ctx_win+1:, :]
    E, e = equity(t_a, Ret_f.squeeze(-1), cfg['data']['actions']['commission_value'])
    print(f'Strategy equity: {e.detach().cpu().numpy().item():.4f}')
    opt_act = optimal_allocation(Ret_f.squeeze(-1), c = cfg['data']['actions']['commission_value'], loss_fn = loss_fn)
    E, e = equity(opt_act, Ret_f.squeeze(-1), cfg['data']['actions']['commission_value'])
    print(f'Optimal equity: {e.detach().cpu().numpy().item():.4f}')


    