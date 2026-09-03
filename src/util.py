import hashlib
import json
import logging
import os
import sys

import torch


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def truncate(x, max_len):
    if x.size(1) >= max_len:
        x = x[:, -max_len:, :]
    return x


def set_logger(cfg):
    logger = logging.getLogger(cfg['experiment_name'])
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


# --------------------------------------------------------------------------
# Collapse diagnostics
# --------------------------------------------------------------------------

@torch.no_grad()
def latent_diagnostics(Z):
    """Cheap collapse detectors for the encoder output.

    Z: [B, T, D]. Returns per-dimension std (collapses to ~0), and the effective
    rank of the latent covariance -- exp of the entropy of the normalised
    eigenvalue spectrum. A healthy D-dim latent sits near D; a collapsed one
    falls to ~1. Log both, since SIGReg can keep the marginals Gaussian while
    the code still lives on a low-dimensional subspace.
    """
    Zf = Z.reshape(-1, Z.size(-1)).float()
    std = Zf.std(dim=0).mean()

    Zc = Zf - Zf.mean(dim=0, keepdim=True)
    cov = (Zc.T @ Zc) / max(Zc.size(0) - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    total = eigvals.sum()

    if total <= 0:
        return {'latent_std': std, 'latent_erank': torch.zeros((), device=Z.device)}

    p = eigvals / total
    entropy = -(p * torch.log(p + 1e-12)).sum()
    return {'latent_std': std, 'latent_erank': torch.exp(entropy)}


# --------------------------------------------------------------------------
# Discretisation
# --------------------------------------------------------------------------

def preprocess_classes(data, min_value, max_value, num_bins):
    """Clip to [min_value, max_value] and assign uniform bin indices.

    Returns (clipped_values, bin_indices). The continuous value is clipped too,
    so the AdaLN condition and the CE target describe the same quantity -- an
    unclipped condition next to a clipped target is a mismatch the model has to
    undo.
    """
    B, Seq = data.shape
    data = data.view(B, Seq, 1).float().clamp(min_value, max_value)

    data_flat = data.reshape(B * Seq)

    bin_edges = torch.linspace(min_value, max_value + 1e-6, num_bins + 1)
    idx = torch.bucketize(data_flat, bin_edges)
    idx = torch.clamp(idx - 1, 0, num_bins - 1)

    return data, idx.view(B, Seq, 1).long()


def bin_centers(min_value, max_value, num_bins):
    """Centres of the bins produced by preprocess_classes.

    linspace(min, max, num_bins) gives edges, not centres; using it as the
    decode table biases every reconstructed value by half a bin.
    """
    edges = torch.linspace(min_value, max_value, num_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


# --------------------------------------------------------------------------
# Equity / objectives
# --------------------------------------------------------------------------

def delta_equity(x, p, c):
    # x is [B, seq_len + 1], p is [B, seq_len]
    x_H = x[:, 1:]
    x_d = torch.abs(torch.diff(x, dim=-1))

    E = x_H * p - c * x_d + 1.
    return E


def equity(x, p, c):
    dE = delta_equity(x, p, c)
    E = torch.cumprod(dE, dim=-1)
    return E, E[:, -1]


def loss_fn_eq(x, p, c):
    return -torch.sum(torch.sum(torch.log(delta_equity(x, p, c) + 1e-6), dim=-1))


def loss_fn_so(x, p, c):
    dE = delta_equity(x, p, c)

    # Downside deviation is measured against the target return (here 0 excess,
    # i.e. dE == 1), not against the mean of the negative values.
    shortfall = torch.clamp(1.0 - dE, min=0.0)
    downside_std = torch.sqrt(torch.mean(shortfall ** 2, dim=-1) + 1e-12)

    sortino = (torch.mean(dE, dim=-1) - 1.0) / (downside_std + 1e-6)

    return -torch.sum(sortino)


def loss_fn_sh(x, p, c):
    dE = delta_equity(x, p, c)
    total_std = torch.std(dE, dim=-1, unbiased=True)
    sharpe = (torch.mean(dE, dim=-1) - 1.0) / (total_std + 1e-6)
    return -torch.sum(sharpe)


# Selectable via `[data.actions.loss_fn]`; keys are the config values.
LOSS_FNS = {
    'equity': loss_fn_eq,
    'sortino': loss_fn_so,
    'sharpe': loss_fn_sh,
}


def make_constrained_loss(base_loss_fn, max_change=0.1, penalty_weight=1000.0):
    def constrained_loss_fn(x, p, c):
        original_loss = base_loss_fn(x, p, c)
        x_diff = torch.abs(torch.diff(x, dim=-1))
        excess = torch.clamp(x_diff - max_change, min=0.0)
        penalty = penalty_weight * torch.sum(excess ** 2)

        return original_loss + penalty

    return constrained_loss_fn


def optimal_allocation(p, c, x0=0.0, loss_fn=loss_fn_eq, lr=0.01, steps=1_000):
    if p.dim() == 1:
        p = p.unsqueeze(0)

    B, seq_len = p.shape

    if not isinstance(x0, torch.Tensor):
        x0 = torch.full((B, 1), x0, dtype=p.dtype, device=p.device)
    else:
        x0 = x0.view(B, 1).to(p.device)

    u_trainable = torch.zeros_like(p, requires_grad=True)
    optimizer = torch.optim.Adam([u_trainable], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        x_trainable = torch.tanh(u_trainable)

        x_full = torch.cat((x0, x_trainable), dim=-1)
        loss = loss_fn(x_full, p, c)

        loss.backward()
        optimizer.step()

    return torch.cat((x0, torch.tanh(u_trainable)), dim=-1).detach()


# --------------------------------------------------------------------------
# Oracle label cache
# --------------------------------------------------------------------------

def cache_key(**kwargs):
    blob = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def load_cached(cache_dir, key):
    path = os.path.join(cache_dir, f'{key}.pt')
    if os.path.exists(path):
        try:
            return torch.load(path, map_location='cpu')
        except Exception:
            return None
    return None


def save_cached(cache_dir, key, tensor):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'{key}.pt')
    tmp = path + '.tmp'
    torch.save(tensor, tmp)
    os.replace(tmp, path)
