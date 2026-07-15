import torch
from omegaconf import OmegaConf
from data import load_data

if __name__ == '__main__':
    cfg = OmegaConf.load('./config.yaml')
    data, targets = load_data(cfg)

    X = torch.from_numpy(data[0])
    print(X.shape)
    X = X.unfold(0, cfg['data']['window_size'], cfg['data']['window_size']).transpose(1,2)
    print(X.shape)
    X = X.unfold(0, cfg['data']['sequence_length'], cfg['data']['stride']).permute(0, 3, 1, 2)
    print(X.shape)

    print(X[0, :].shape)
    print('-'*32)

    y = torch.from_numpy(targets[0])
    print(y.shape)
    y = (y + 1.).unfold(0, cfg['data']['window_size'], cfg['data']['window_size'])
    print(y.shape)
    y = (y.cumprod(1) - 1.)[:,-1]
    print(y.shape)
    y = y.unfold(0, cfg['data']['sequence_length'], cfg['data']['stride'])
    print(y.shape)

    print(y.size(0))
