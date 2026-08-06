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

    n_runs = 1
    horizon = 20
    forward_look = 20
    temperature = 0.4
    re_try = 1

    cfg = OmegaConf.load('./config.yaml')
    logger = set_logger(cfg)
    jepa_path = f'./models/jepa/{cfg['jepa']['name']}/last.ckpt'
    model = JEPA.load_from_checkpoint(jepa_path, cfg=cfg, weights_only=False)
    model.eval()
    model.cuda()
    logger.info(f'Loaded JEPA model from: {jepa_path}')

    test_dataset = CryptoDataset(cfg, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    for _ in range(n_runs):
        batch = next(iter(test_loader))
        X = batch['sample'].cuda()       # [1, Seq, Win, D]
        y_true = batch['target'].cuda()  # [1, Seq, 1] (Fra det nye data-setup)
        ret_true = batch['return'].cuda() # [1, Seq, 1] (De sande float-afkast)

        start_idx = X.size(1) - horizon - 1

        ps = 0.0
        positions_size = [ps]
        returns = []

        with torch.no_grad():
            Z = model.encode(X)

        for t in range(horizon):
            Z_history = Z[:, :start_idx+1+t, :]          # [1, T_hist, d_model]
            Ret_history = ret_true[:, :start_idx+1+t, :]      # [1, T_hist, 1]

            ps_temp = []
            for _ in range(re_try):
                with torch.no_grad():
                    states, logits, ret = model.imagine(Z_history, Ret_history, horizon = forward_look, temperature = temperature)
                ret = ret.squeeze(0).squeeze(-1).detach().cpu()
                U = optimal_allocation(p = ret, c = 0.0005, x0 = ps, loss_fn = loss_fn_sh)
                ps_ = float(U[1].detach().cpu())
                ps_temp.append(ps_)
            ps = np.mean(ps_temp)
            positions_size.append(ps)
            returns.append(float(ret_true[:, start_idx+1+t, :].detach().cpu()))

        print('positions_size', positions_size)
        print('returns', returns)
        positions_size = torch.tensor(positions_size)
        returns = torch.tensor(returns)
        E, e, = equity(positions_size, returns, 0.0005)
        print('Equity', E)
        print('Final Equity: ', e)


