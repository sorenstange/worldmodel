import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def delta_equity(x, p, c):
    x_H = x[1:]
    x_d = torch.abs(torch.diff(x))

    E = x_H * p - c * x_d + 1.

    return E

def equity(x, p, c):
    dE = delta_equity(x, p, c)
    E = torch.cumprod(dE, dim = -1)
    return E, E[-1]

def loss_fn(x, p, c):
    return -torch.sum(torch.log(delta_equity(x, p, c) + 1e-6))

def optimal_allocation(p, c, x0 = 0.0, lr = 0.01, steps = 1_000):
    h = len(p)
    x0 = torch.tensor([x0], requires_grad=False)
    u_trainable = torch.zeros_like(p, requires_grad=True)
    optimizer = torch.optim.Adam([u_trainable], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        x_trainable = torch.tanh(u_trainable)
        
        x_full = torch.cat((x0, x_trainable))
        loss = loss_fn(x_full, p, c)

        loss.backward()
        optimizer.step()
    
    x = torch.cat((x0, torch.tanh(u_trainable)))

    return x

if __name__ == '__main__':
    torch.manual_seed(42)

    h = 15
    sigma = 0.01
    c = 0.0005
    
    p = sigma * torch.randn(h)
    x = optimal_allocation(p, c)

    print('x:', x[1:].detach().numpy())
    print('p:', p.numpy())

    E, E_h = equity(x, p, c)
    print('E:', E.detach().numpy())
    print('E_h: ', float(E_h.detach()))
