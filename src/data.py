#data.py
import os
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from binance.client import Client
from binance.enums import *

from util import *

FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Volatility']
# Floor on the volatility normaliser, in 1m return units. Guards against a dead
# or halted market turning a normal return into a huge z-score.
VOL_FLOOR = 1e-5


class CryptoDataset(Dataset):
    """Windowed sequences of normalised candles plus oracle allocation labels.

    Two different return series are carried through deliberately:
      * `return_raw`  -- the compounded 15m return, in real units. Everything
                         economic (equity, commission, oracle labels) uses this.
      * `return_norm` -- `return_raw` divided by a causal volatility estimate.
                         This is what the model conditions on and predicts, so
                         that BTC and SOL are on one scale (SOL's 15m sigma is
                         ~1.9x BTC's -- see CLAUDE.md).
    """

    def __init__(self, cfg, mode='training', make_action=False):
        logger = logging.getLogger(cfg['experiment_name'])

        dcfg = cfg['data']
        W = dcfg['window_size']
        S = dcfg['sequence_length']
        # Training windows overlap as cheap augmentation; evaluation windows must
        # not, or the reported metrics average correlated duplicates.
        stride = dcfg['stride'] if mode == 'training' else dcfg.get('eval_stride', S)

        logger.info(f'Initializing CryptoDataset. Mode: {mode}. Window size: {W}. Stride: {stride}')

        self.samples, self.raw_returns, self.vols, self.actions = [], [], [], []
        # Which symbol each sequence came from, so evaluation can break results
        # down per market rather than reporting one pooled number.
        self.symbol_names, self.symbol_ids = [], []

        loss_fn = make_constrained_loss(
            loss_fn_so,
            max_change=dcfg['actions']['max_change'],
            penalty_weight=dcfg['actions']['penalty_weight'],
        )

        for symbol, d, t, v, fingerprint in load_data(cfg, mode):
            d = torch.from_numpy(d)
            t = torch.from_numpy(t)
            v = torch.from_numpy(v)

            # [T, F] -> [N, W, F] -> [M, S, W, F]
            d = d.unfold(0, W, W).transpose(1, 2)
            d = d.unfold(0, S, stride).permute(0, 3, 1, 2)

            # 1m returns -> compounded per-window return -> [M, S]
            t = (t + 1.).unfold(0, W, W)
            t = (t.cumprod(1) - 1.)[:, -1]
            t = t.unfold(0, S, stride)

            # Volatility normaliser, taken at the first bar of each window. That
            # value is computed from data strictly before the window starts, so
            # it is known at decision time.
            v = v.unfold(0, W, W)[:, 0] * (W ** 0.5)
            v = v.unfold(0, S, stride).clamp(min=VOL_FLOOR * (W ** 0.5))

            if make_action:
                a = self._oracle_actions(cfg, symbol, mode, t, fingerprint, loss_fn, logger)
                self.actions.append(a)

            self.symbol_ids.append(torch.full((d.size(0),), len(self.symbol_names), dtype=torch.long))
            self.symbol_names.append(symbol)

            self.samples.append(d)
            self.raw_returns.append(t)
            self.vols.append(v)

        if not self.samples:
            raise RuntimeError(
                f"No data for mode '{mode}'. Check data/processed/*.csv and the "
                f"'{mode}_interval' in config.yaml."
            )

        self.samples = torch.concatenate(tuple(self.samples))
        self.raw_returns = torch.concatenate(tuple(self.raw_returns))
        self.vols = torch.concatenate(tuple(self.vols))
        self.symbol_ids = torch.concatenate(tuple(self.symbol_ids))

        norm_returns = self.raw_returns / self.vols
        self.returns, self.return_targets = preprocess_classes(
            norm_returns,
            dcfg['returns']['min_value'],
            dcfg['returns']['max_value'],
            dcfg['returns']['num_bins'],
        )
        self.raw_returns = self.raw_returns.unsqueeze(-1)
        self.vols = self.vols.unsqueeze(-1)

        if make_action:
            self.actions = torch.concatenate(tuple(self.actions))
            self.actions, self.action_targets = preprocess_classes(
                self.actions,
                dcfg['actions']['min_value'],
                dcfg['actions']['max_value'],
                dcfg['actions']['num_bins'],
            )
        else:
            self.actions = None
            self.action_targets = None

        clipped = (norm_returns.abs() > dcfg['returns']['max_value']).float().mean()
        logger.info(
            f"Dataset created! Mode: {mode}. Sequences: {self.samples.size(0):,}. "
            f"Clipped returns: {100 * clipped:.3f}%"
        )

    def _oracle_actions(self, cfg, symbol, mode, t, fingerprint, loss_fn, logger):
        """Oracle labels are ~1000 Adam steps per symbol, so cache them."""
        dcfg = cfg['data']
        key = cache_key(
            symbol=symbol, mode=mode, fingerprint=fingerprint,
            window=dcfg['window_size'], seq=dcfg['sequence_length'],
            stride=dcfg['stride'] if mode == 'training' else dcfg.get('eval_stride', dcfg['sequence_length']),
            commission=dcfg['actions']['commission_value'],
            max_change=dcfg['actions']['max_change'],
            penalty=dcfg['actions']['penalty_weight'],
            steps=dcfg['actions']['opt_steps'], lr=dcfg['actions']['opt_lr'],
            loss='sortino_v2',
        )
        cache_dir = './data/cache'

        cached = load_cached(cache_dir, key)
        if cached is not None and cached.shape == t.shape:
            logger.info(f'  {symbol}: oracle labels loaded from cache ({key})')
            return cached

        logger.info(f'  {symbol}: optimising oracle allocations for {t.size(0):,} sequences...')
        a = optimal_allocation(
            t,
            dcfg['actions']['commission_value'],
            x0=0.0,
            loss_fn=loss_fn,
            lr=dcfg['actions']['opt_lr'],
            steps=dcfg['actions']['opt_steps'],
        )[:, 1:]
        save_cached(cache_dir, key, a)
        return a

    def __len__(self):
        return self.samples.size(0)

    def __getitem__(self, idx):
        item = {
            'sample': self.samples[idx],
            'return': self.returns[idx],
            'return_target': self.return_targets[idx],
            'return_raw': self.raw_returns[idx],
            'vol': self.vols[idx],
            'symbol_id': self.symbol_ids[idx],
        }
        if self.actions is not None:
            item['action'] = self.actions[idx]
            item['action_target'] = self.action_targets[idx]
        return item


def load_data(cfg, mode='training'):
    logger = logging.getLogger(cfg['experiment_name'])
    dcfg = cfg['data']

    for symbol in dcfg['symbols']:
        path = f'./data/processed/{symbol}.csv'
        if not os.path.exists(path):
            logger.warning(f'{symbol}: no processed file at {path}, skipping.')
            continue

        df = pd.read_csv(path)
        df.set_index('OpenTime', inplace=True)

        lo, hi = dcfg[f'{mode}_interval']
        df = df[(df.index >= lo) & (df.index < hi)]
        if len(df) < dcfg['window_size'] * dcfg['sequence_length']:
            logger.warning(f'{symbol}: too few rows in {mode} interval ({len(df)}), skipping.')
            continue

        # Identifies this exact slice, so the oracle cache invalidates when the
        # underlying CSV is extended or re-preprocessed.
        fingerprint = f'{len(df)}:{df.index[0]}:{df.index[-1]}'

        yield (
            symbol,
            df[FEATURE_COLS].values.astype(np.float32),
            df['Return'].values.astype(np.float32),
            df['RetVol'].values.astype(np.float32),
            fingerprint,
        )


def update_data(cfg):
    logger = logging.getLogger(cfg['experiment_name'])
    logger.info('Updating data...')
    cfg = cfg['data']
    tf = cfg['timeframe']
    symbols = cfg['symbols']

    start_date = min([min(cfg[period]) for period in ['training_interval', 'validation_interval', 'test_interval']])
    end_date = max([max(cfg[period]) for period in ['training_interval', 'validation_interval', 'test_interval']])
    os.makedirs('./data/raw', exist_ok=True)

    for symbol in symbols:
        try:
            path = f'./data/raw/{symbol}.csv'
            if not os.path.exists(path):
                logger.info(f'Data for {symbol} {tf} does not exist... Downloading data...')
                df = get_OHLCV(symbol, tf, SINCE=start_date, TO=end_date)
                df.to_csv(path, index=False)
                logger.info(f'Data for {symbol} {tf} downloaded to {path}.')
            else:
                df = pd.read_csv(path)
                current_end_date = max(df['OpenTime'])
                df['OpenTime'] = pd.to_datetime(df['OpenTime'])

                logger.info(f'Updating data for {symbol} {tf}...')
                df_updated = get_OHLCV(symbol, tf, SINCE=current_end_date, TO=end_date)
                df_updated['OpenTime'] = pd.to_datetime(df_updated['OpenTime'])
                df = pd.concat([df, df_updated])
                df.drop_duplicates(subset=['OpenTime'], keep='last', inplace=True)
                df.to_csv(path, index=False)
                logger.info(f'Data for {symbol} {tf} downloaded to {path}!')
        except Exception as e:
            logger.error(f'Error at {symbol}: {e}')

    logger.info('Update complete!')


def compute_features(df, cfg):
    """Causal feature construction. Shared by preprocess_data and live inference
    so the two can never drift apart."""
    win = cfg['normalization_window']
    short = cfg.get('vol_window_short', 60)

    df['Return'] = df['Close'].pct_change()
    log_ret = np.log1p(df['Return'])

    mu = df['Close'].rolling(window=win).mean()
    std = df['Close'].rolling(window=win).std()
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = (df[col] - mu) / (std + 1e-8)

    df['Volume'] = np.log1p(df['Volume'])
    df['Volume'] = (
        (df['Volume'] - df['Volume'].rolling(window=win).mean())
        / (df['Volume'].rolling(window=win).std() + 1e-8)
    )

    # Realised volatility of returns. rv_long is the normaliser for the return
    # target; the model feature is the short/long ratio, which is scale-free and
    # therefore comparable across symbols (a raw std is not).
    rv_long = log_ret.rolling(window=win).std()
    rv_short = log_ret.rolling(window=short).std()
    df['Volatility'] = np.log((rv_short + 1e-8) / (rv_long + 1e-8))

    # Shifted by one bar so the normaliser for a window is known strictly before
    # that window begins.
    df['RetVol'] = rv_long.shift(1)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df[FEATURE_COLS + ['Return', 'RetVol']].copy()


def preprocess_data(cfg):
    logger = logging.getLogger(cfg['experiment_name'])
    cfg = cfg['data']
    logger.info('Preprocessing data...')
    tf = cfg['timeframe']
    symbols = cfg['symbols']
    os.makedirs('./data/processed', exist_ok=True)

    for symbol in symbols:
        try:
            logger.info(f'Preprocessing {symbol} {tf}')
            path = f'./data/raw/{symbol}.csv'
            df = pd.read_csv(path)
            df.set_index('OpenTime', inplace=True)
            df.sort_index(inplace=True)

            df = compute_features(df, cfg)
            df.to_csv(f'./data/processed/{symbol}.csv')
        except Exception as e:
            logger.error(f'Error at {symbol}: {e}')

    logger.info('Preprocessing complete!')


client = Client()


def get_OHLCV(SYMBOL: str, TIMEFRAME: str, SINCE: str = '2018-01-01 00:00', TO: str = None):
    df = pd.DataFrame(
        client.get_historical_klines(
            SYMBOL,
            TIMEFRAME,
            start_str=SINCE,
            end_str=TO,
            klines_type=HistoricalKlinesType.FUTURES
        )
    )

    df.columns = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
                  'CloseTime', 'QouteAssetVolume', 'NumberOfTrades',
                  'TakerBuyBaseAssetVolume', 'TakerBuyQouteAssetVolumne',
                  'Ignore']

    df.OpenTime = pd.to_datetime(df.OpenTime, unit='ms')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype('float')

    df.dropna(axis=0, inplace=True)
    return df[['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume']]


def main(cfg):
    logger = set_logger(cfg)
    logger.info('Starting data pipeline')
    update_data(cfg)
    preprocess_data(cfg)


if __name__ == '__main__':
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.load('./config.yaml')
    cfg = OmegaConf.to_container(cfg, resolve=True)
    main(cfg)
