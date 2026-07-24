import torch
import torch.nn.functional as F

if __name__ == '__main__':
    X = torch.rand((64, 15))
    s = X.sum(dim=-1)
    print(s)
    print(s.shape)
    a = s.mean(dim=-1)
    print(a)
    print(a.shape)

