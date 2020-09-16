import logging
from lib.log import logger
import pandas as pd
from old.lib.plotter import correlation
import lib.dataset as builder
import json

INTERACTIVE_FIGURE = False
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
    'IOT',
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
    filename='../dataset_ohlcv_social.log',
    filemode='w',
    root_level=logging.DEBUG,
    log_level=logging.DEBUG,
    logger='dataset ohlcv_social'
)

index = {}
for _sym in SYMBOLS:
    ohlcv = pd.read_csv("./data/preprocessed/ohlcv/csv/{}.csv".format(_sym.lower()), sep=',', encoding='utf-8',
                        index_col='Date', parse_dates=True)
    cm = pd.read_csv("./data/preprocessed/cryptocompare_social/csv/{}.csv".format(_sym.lower()), sep=',', encoding='utf-8',
                     index_col='Date', parse_dates=True)
    cm_pct = builder.pct_change(cm)
    df = pd.concat([ohlcv, cm_pct], axis='columns', verify_integrity=True, sort=True, join='inner')
    df['target'] = builder.target_discrete_price_variation(ohlcv, periods=1)
    df['target_pct'] = builder.target_price_variation(ohlcv, periods=1)
    df['target_label'] = builder.target_discrete_price_variation(ohlcv, periods=1, labels=True)
    df = df[df['close'].notna()] # Only keep rows where close is not na
    #
    csv_path = 'data/datasets/ohlcv_social/csv/{}.csv'.format(_sym.lower())
    xls_path = 'data/datasets/ohlcv_social/excel/{}.xlsx'.format(_sym.lower())
    df.to_csv(csv_path, sep=',', encoding='utf-8', index=True, index_label='Date')
    df.to_excel(xls_path, index=True, index_label='Date')
    corr = df.corr()
    correlation(corr, 'data/datasets/ohlcv_social/correlation/{}.png'.format(_sym.lower()), title='{} OHLCV and CryptoCompare Social data'.format(_sym), figsize=(40,30))
    corr.to_excel('data/datasets/ohlcv_social/correlation/{}.xlsx'.format(_sym.lower()), index=True, index_label='Date')
    index[_sym] = {'csv':csv_path, 'xls':xls_path}

    print('Saved {} in data/datasets/ohlcv_social/'.format(_sym))

with open('data/datasets/ohlcv_social/index.json', 'w') as f:
    json.dump(index, f, sort_keys=True, indent=4)