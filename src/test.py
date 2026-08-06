import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from util import *
from data import get_OHLCV
import pandas as pd
import time

def make_constrained_loss(base_loss_fn, max_change=0.1, penalty_weight=1000.0):
    def constrained_loss_fn(x, p, c):
        original_loss = base_loss_fn(x, p, c)
        x_diff = torch.abs(torch.diff(x, dim=-1))
        overskridelse = torch.clamp(x_diff - max_change, min=0.0)
        penalty = penalty_weight * torch.sum(overskridelse ** 2)
        
        return original_loss + penalty
        
    return constrained_loss_fn

if __name__ == '__main__':
    SYMBOL = 'BTCUSDT'
    TIMEFRAME = '15m'
    c = 0.0005
    df = get_OHLCV(SYMBOL, TIMEFRAME, SINCE = '2025-08-01 00:00', TO = '2026-08-01 00:00')

    p = torch.from_numpy(df['Close'].pct_change().dropna().values.astype(np.float32))
    p = p.unfold(0, 64, 32)

    loss_fn = make_constrained_loss(loss_fn_so)
    start = time.time()
    x = optimal_allocation(p, c, x0 = 0.0, loss_fn = loss_fn, lr = 0.01, steps = 1_000)
    x = x.detach()
    print(f'Time used: {time.time() - start:.2f} sec')
    E, e = equity(x, p, c)
    x = x.detach()
    E = E.detach()
    e = e.detach()

    print('Mean final equity: ', torch.mean(e))

    x_values = x[:, 1:].detach().cpu().numpy().flatten()

    # Opret plottet
    plt.figure(figsize=(10, 6))
    
    # Tegn histogrammet (her med 50 søjler for god detalje)
    plt.hist(x_values, bins=61, edgecolor='black', alpha=0.7, color='royalblue')
    
    # Tilføj titler og labels
    plt.title(f'Histogram over Allokeringer (x) for {SYMBOL}', fontsize=14)
    plt.xlabel('Allokeringsværdi (tanh output)', fontsize=12)
    plt.ylabel('Antal observationer (frekvens)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Gem figuren i den ønskede mappe
    output_path = './figs/histogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close() # Luk figuren for at frigive hukommelse

    print(f'Histogrammet er blevet gemt succesfuldt i: {output_path}')
    actions = x[:, 1:].detach()
    print('actions.shape: ', actions.shape)

    actions, action_targets = preprocess_classes(actions, -1.0, 1.0, 51)
    print('actions: ', actions)
    print('action_targets: ', action_targets)