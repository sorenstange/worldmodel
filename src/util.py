import torch
import logging
import sys

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

def set_logger(cfg):
    logger = logging.getLogger(cfg['experiment_name'])
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # Tilføj tidsstempel (asctime) foran beskeden (message)
        # datefmt bestemmer hvordan tiden ser ud (f.eks. 14:30:05)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(message)s", 
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False 
    
    return logger

def preprocess_classes(data, min_value, max_value, num_bins):
    B, Seq = data.shape
    data = data.view(B, Seq, 1).float()

    data_flat = data.reshape(B * Seq)
    data_flat = data_flat.clamp(min_value, max_value)

    bin_edges = torch.linspace(min_value, max_value + 1e-6, num_bins + 1)
    data_flat = torch.bucketize(data_flat, bin_edges)
    data_flat = torch.clamp(data_flat - 1, 0, num_bins - 1)

    data_targets = data_flat.view(B, Seq, 1).long()
    return data, data_targets


def delta_equity(x, p, c):
    # x har dim [B, seq_len + 1], p har dim [B, seq_len]
    x_H = x[:, 1:]
    x_d = torch.abs(torch.diff(x, dim=-1))

    E = x_H * p - c * x_d + 1.
    return E

def equity(x, p, c):
    dE = delta_equity(x, p, c)
    E = torch.cumprod(dE, dim=-1)
    return E, E[:, -1]

def loss_fn_eq(x, p, c):
    # Summerer log-afkast over tid, og summerer uafhængigt over batchen
    return -torch.sum(torch.sum(torch.log(delta_equity(x, p, c) + 1e-6), dim=-1))

def loss_fn_so(x, p, c):
    dE = delta_equity(x, p, c)
    
    # Maske for negative værdier [B, seq_len]
    neg_mask = (dE < 0).float()
    neg_dE = dE * neg_mask
    
    # Antal negative elementer per batch
    n_neg = torch.sum(neg_mask, dim=-1, keepdim=True)
    
    # Gennemsnit af de negative værdier per batch
    mean_neg = torch.sum(neg_dE, dim=-1, keepdim=True) / (n_neg + 1e-6)
    
    # Kvadreret afvigelse (kun for de negative elementer)
    sq_deviation = (neg_dE - mean_neg) ** 2 * neg_mask
    
    # Downside varians og standardafvigelse per batch
    variance = torch.sum(sq_deviation, dim=-1) / torch.clamp(n_neg.squeeze(-1) - 1, min=1)
    downside_std = torch.sqrt(variance + 1e-6)
    
    # Hvis en batch har <= 1 negativ værdi, sæt downside_std til 0
    downside_std = torch.where(n_neg.squeeze(-1) > 1, downside_std, torch.zeros_like(downside_std))
        
    # Beregn Sortino ratio per trajectory
    sortino = torch.mean(dE, dim=-1) / (downside_std + 1e-6)
    
    # Returner minus summen for uafhængig optimering
    return -torch.sum(sortino)

def loss_fn_sh(x, p, c):
    dE = delta_equity(x, p, c)
    
    # Standardafvigelse for HELE afkast-sekvensen per batch
    # Vi bruger unbiased=True (bessels korrektion) for at matche torch.std adfærd
    total_std = torch.std(dE, dim=-1, unbiased=True)
    
    # Beregn Sharpe ratio per trajectory
    sharpe = torch.mean(dE, dim=-1) / (total_std + 1e-6)
    
    # Returner minus summen for uafhængig optimering
    return -torch.sum(sharpe)

def make_constrained_loss(base_loss_fn, max_change=0.1, penalty_weight=1000.0):
    def constrained_loss_fn(x, p, c):
        original_loss = base_loss_fn(x, p, c)
        x_diff = torch.abs(torch.diff(x, dim=-1))
        overskridelse = torch.clamp(x_diff - max_change, min=0.0)
        penalty = penalty_weight * torch.sum(overskridelse ** 2)
        
        return original_loss + penalty
        
    return constrained_loss_fn

def optimal_allocation(p, c, x0 = 0.0, loss_fn = loss_fn_eq, lr = 0.01, steps = 1_000):
    if p.dim() == 1:
        p = p.unsqueeze(0)
        
    # p har dim [B, seq_len]
    B, seq_len = p.shape
    
    # Opret x0 med korrekt batch-dimension [B, 1]
    if not isinstance(x0, torch.Tensor):
        x0 = torch.full((B, 1), x0, dtype=p.dtype, device=p.device)
    else:
        x0 = x0.view(B, 1).to(p.device)
        
    u_trainable = torch.zeros_like(p, requires_grad=True)
    optimizer = torch.optim.Adam([u_trainable], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        x_trainable = torch.tanh(u_trainable)
        
        # Sammenføj langs sekvens-dimensionen (dim=-1) -> [B, seq_len + 1]
        x_full = torch.cat((x0, x_trainable), dim=-1)
        loss = loss_fn(x_full, p, c)

        loss.backward()
        optimizer.step()
    
    x = torch.cat((x0, torch.tanh(u_trainable)), dim=-1)

    return x
