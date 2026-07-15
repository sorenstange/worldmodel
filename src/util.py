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

def gaussian_label_smoothing(y_binned, num_bins, sigma=1.0):
    """
    y_binned: Tensor med bin-indices (shape: [batch_size])
    num_bins: Int, antallet af bins
    sigma: Float, styrken af din smoothing (hvor bred skal klokkekurven være)
    """
    # 1. Lav en vektor med alle mulige bin-indices: [0, 1, 2, ..., num_bins-1]
    bin_indices = torch.arange(num_bins, device=y_binned.device).float()
    
    # 2. Udregn den absolutte afstand fra det sande bin til alle andre bins (broadcasting)
    # Shape bliver: [batch_size, num_bins]
    distances = bin_indices.unsqueeze(0) - y_binned.unsqueeze(1).float()
    
    # 3. Anvend Gauss-formlen for at udregne u-normerede sandsynligheder
    smooth_targets = torch.exp(-0.5 * (distances / sigma) ** 2)
    
    # 4. Normer rækkerne så de summerer til 1.0 (vigtigt for KL-Divergence)
    smooth_targets = smooth_targets / smooth_targets.sum(dim=1, keepdim=True)
    
    return smooth_targets