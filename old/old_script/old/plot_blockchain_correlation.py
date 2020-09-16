import logging
from lib.log import logger
from lib.symbol import Symbol, DatasetType

import pandas as pd
from old.lib.plotter import correlation

SYMBOLS = [
    'ADA',
    'BCH',
    'BNB',
    'BTC',
    'BTG',
    'DASH',
    'DOGE',
    'EOS',
    'ETC',
    'ETH',
    'LTC',
    'LINK',
    'NEO',
    'QTUM',
    'TRX',
    'USDT',
    'VEN',
    'WAVES',
    'XEM',
    'XMR',
    'XRP',
    'ZEC',
    'ZRX'
]

logger.setup(
    filename='../job_test.log',
    filemode='w',
    root_level=logging.DEBUG,
    log_level=logging.DEBUG,
    logger='job_test'
)

ohlcv = pd.read_csv("./data/result/ohlcv.csv", sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
chain = pd.read_csv("./data/result/blockchains.csv", sep=',', encoding='utf-8', index_col='Date', parse_dates=True)

for _sym in SYMBOLS:
    s = Symbol(_sym, ohlcv=ohlcv, blockchain=chain[[c for c in chain.columns if c.startswith(_sym)]], column_map={
        'open': _sym+'_Open',
        'high': _sym+'_High',
        'low': _sym+'_Low',
        'close': _sym,
        'volume': _sym+'_Volume'
    })

    bchn = s.get_dataset(DatasetType.BLOCKCHAIN)
    correlation(bchn.corr(), 'data/result/blockchain-{}-corr.png'.format(_sym), title="{} data from CoinMetrics and OHLCV".format(_sym), figsize=(32,18))
    #pat = s.get_dataset(DatasetType.OHLCV_PATTERN)
    #bchn.to_csv('data/result/block1chain-dataset.csv', sep=',', encoding='utf-8', index=True, index_label='Date')
  #  pat.to_csv('data/result/ohlcv_pattern.csv', sep=',', encoding='utf-8', index=True, index_label='Date')
    #fourier_transform(bchn['CapMVRVCur'])


