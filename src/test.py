import torch
from omegaconf import OmegaConf
from data import CryptoDataset
from util import set_logger

if __name__ == '__main__':
    cfg = OmegaConf.load('./config.yaml')
    logger = set_logger(cfg)

    dataset = CryptoDataset(cfg)
