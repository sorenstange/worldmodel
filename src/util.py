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