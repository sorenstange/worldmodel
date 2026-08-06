import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from util import *
from data import get_OHLCV
import pandas as pd

if __name__ == '__main__':
    SYMBOL = 'BTCUSDT'
    TIMEFRAME = '15m'
    c = 0.0005
    df = get_OHLCV(SYMBOL, TIMEFRAME, SINCE = '2026-08-05 00:00', TO = '2026-08-06 00:00')

    p = torch.from_numpy(df['Close'].pct_change().dropna().values.astype(np.float32))

    x = optimal_allocation(p, c, x0 = 0.0, loss_fn = loss_fn_so, lr = 0.01, steps = 1_000)
    E, e = equity(x, p, c)

    print('p: ', p)
    print('x: ', x)
    print('E: ', E)
    print('Final Equity: ', e)
