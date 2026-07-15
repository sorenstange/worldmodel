#data.py
import os
import pandas as pd
import numpy as np
import logging

import torch
from torch.utils.data import Dataset

from binance.client import Client
from binance.enums import *

class CryptoDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        logger = logging.getLogger(cfg['experiment_name'])
        logger.info(f'Initializing CryptoDataset. Mode: {mode}. Window size: {cfg['data']['lookback_window_size']}')
        self.window_size    = cfg['data']['lookback_window_size']

        self.samples        = []
        self.targets        = []
        self.datasets_start = []
        self.start_indices  = []

        data, targets = load_data(cfg, mode)

        n_samples = 0
        for d, t in zip(data, targets):
            idx = np.arange(self.window_size, d.shape[0])
            self.start_indices.append(idx)
            self.datasets_start.append(n_samples)
            n_samples += len(idx)

            for j in idx:
                self.samples.append(torch.FloatTensor(d[j - self.window_size:j, :]))
                self.targets.append(torch.FloatTensor([t[j]]))
                
        logger.info(f"Dataset created! Mode: {mode}. Number of points: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {'sample' : self.samples[idx], 'target' : self.targets[idx]}

def load_data(cfg, mode = 'train'):
    cfg     = cfg['data']
    tf      = cfg['timeframe']
    symbol  = cfg['symbol']

    data, targets = [], []

    df = pd.read_csv(f'./data/processed/{tf}/{symbol}.csv')
    df.set_index('OpenTime', inplace = True)
  
    df = df[((df.index >= cfg[f'{mode}_interval'][0]) & (df.index < cfg[f'{mode}_interval'][1]))]

    data.append(df.drop('Return', axis = 1).values.astype(np.float32))
    targets.append(df['Return'].shift(-1).fillna(0).values.astype(np.float32))

    return data, targets

def update_data(cfg):
    logger = logging.getLogger(cfg['experiment_name'])
    logger.info('Updating data...')
    cfg = cfg['data']
    tf  = cfg['timeframe']
    symbols = cfg['symbol']

    start_date  = min([min(cfg[period]) for period in ['train_interval', 'val_interval', 'test_interval']])
    end_date    = max([max(cfg[period]) for period in ['train_interval', 'val_interval', 'test_interval']])
    os.makedirs(f'./data/raw/{tf}', exist_ok=True)

    for symbol in symbols:
        try:
            path = f'./src/autotrader/LeWM/data/raw/{tf}/{symbol}.csv'
            if not os.path.exists(path): # We need to download the data
                logger.info(f'Data for {symbol} {tf} does not exists... Downloading data...')
                df = get_OHLCV(symbol, tf, SINCE = start_date, TO = end_date)
                df.to_csv(path, index = False) 
                logger.info(f'Data for {symbol} {tf} downloaded to {path}.')
            else:
                df = pd.read_csv(path)
                current_end_date = max(df['OpenTime'])
                df['OpenTime'] = pd.to_datetime(df['OpenTime'])

                logger.info(f'Updating data for {symbol} {tf}...')
                df_updated = get_OHLCV(symbol, tf, SINCE = current_end_date, TO = end_date)
                df_updated['OpenTime'] = pd.to_datetime(df_updated['OpenTime'])
                df = pd.concat([df, df_updated])
                df.drop_duplicates(subset=['OpenTime'], keep = 'last', inplace = True)
                df.to_csv(path, index = False)
                logger.info(f'Data for {symbol} {tf} downloaded to {path}!')
        except Exception as e:
            logger.error(f'Error at {symbol}: {e}')

    logger.info('Update complete!')
    

def preprocess_data(cfg):
    logger = logging.getLogger(cfg['experiment_name'])
    cfg = cfg['data']
    logger.info('Preprocessing data...')
    tf      = cfg['timeframe']
    symbols = cfg['symbol']
    os.makedirs(f'./data/processed/{tf}', exist_ok=True)

    for symbol in symbols:
        try:
            logger.info(f'Preprocessing {symbol} {tf}')
            path = f'./data/raw/{tf}/{symbol}.csv'
            df = pd.read_csv(path)
            df.set_index('OpenTime', inplace = True)
            df.sort_index(inplace = True)

            df['Return'] = df['Close'].pct_change()

            mu = df['Close'].rolling(window = cfg['normalization_window']).mean()
            std = df['Close'].rolling(window = cfg['normalization_window']).std()
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = (df[col] - mu) / (std + 1e-8)

            df['Volume'] = np.log1p(df['Volume'])
            df['Volume'] = (df['Volume'] - df['Volume'].rolling(window = cfg['normalization_window']).mean()) / (df['Volume'].rolling(window = cfg['normalization_window_size']).std() + 1e-8)
            
            df['Volatility'] = std / mu

            df.dropna(inplace=True)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Volatility', 'Return']].copy()

            path = f'./data/processed/{tf}/{symbol}.csv'
            df.to_csv(path)
        except Exception as e:
            logger.error(f'Error at {symbol}: {e}')   

    logger.info('Preprocessing complete!')    

client = Client()

def get_OHLCV(SYMBOL : str, TIMEFRAME : str, SINCE : str = '2018-01-01 00:00', TO : str = None):
    df = pd.DataFrame(
        client.get_historical_klines(
                SYMBOL,
                TIMEFRAME,
                start_str=SINCE,
                end_str = TO,
                klines_type = HistoricalKlinesType.FUTURES
            )
    )

    df.columns = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
                  'CloseTime', 'QouteAssetVolume', 'NumberOfTrades',
                  'TakerBuyBaseAssetVolume', 'TakerBuyQouteAssetVolumne',
                  'Ignore']
    
    df.OpenTime = pd.to_datetime(df.OpenTime, unit = 'ms')
    df.Open     = df.Open.astype('float')
    df.High     = df.High.astype('float')
    df.Low      = df.Low.astype('float')
    df.Close    = df.Close.astype('float')
    df.Volume   = df.Volume.astype('float')

    df.dropna(axis = 0, inplace = True)
    df          = df[['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume']]

    return df

def main(cfg):
    from src.util import set_logger
    logger = set_logger(cfg)
    logger.info('Starting data pipeline')
    update_data(cfg)
    preprocess_data(cfg)

if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('./config.yaml')
    main(cfg)
