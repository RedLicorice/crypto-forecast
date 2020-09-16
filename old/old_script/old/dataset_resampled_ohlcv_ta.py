import logging
from lib.log import logger
import pandas as pd
from old.lib.plotter import correlation
from lib.models import ModelFactory
import lib.dataset as builder
from lib.report import ReportCollection
import json

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
# ModelFactory.discover()
# models = ModelFactory.create_all(['arima','expsmooth','sarima', 'nn', 'kmeans'])

index = {}
for _sym in SYMBOLS:
    ohlcv = pd.read_csv("./data/preprocessed/ohlcv/csv/{}.csv".format(_sym), sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
    #cm = pd.read_csv("./data/preprocessed/coinmetrics.io/csv/{}.csv".format(_sym), sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
    ohlcv_7d = builder.periodic_ohlcv_pct_change(ohlcv, period=7, label=True)
    ohlcv_30d = builder.periodic_ohlcv_pct_change(ohlcv, period=30, label=True)
    ta = builder.features_ta(ohlcv)
    ta_7d = builder.period_resampled_ta(ohlcv, period=7)
    ta_30d = builder.period_resampled_ta(ohlcv, period=30)
    dataframes = [ohlcv, ohlcv_7d, ohlcv_30d, ta, ta_7d, ta_30d]
    for i,df in enumerate(dataframes):
        if df.isnull().values.any():
            logger.error('Null values in dataframe {} for symbol {}'.format(i, _sym))
    df = pd.concat(dataframes, axis='columns', verify_integrity=True, sort=True)
    df['target_pct'] = builder.target_price_variation(ohlcv, periods=1)

    csv_path = 'data/datasets/resampled_ohlcv_ta/csv/{}.csv'.format(_sym.lower())
    xls_path = 'data/datasets/resampled_ohlcv_ta/excel/{}.xlsx'.format(_sym.lower())
    df.to_csv(csv_path, sep=',', encoding='utf-8', index=True, index_label='Date')
    df.to_excel(xls_path, index=True, index_label='Date')
    corr = df.corr()
    correlation(corr, 'data/datasets/resampled_ohlcv_ta/correlation/{}.png'.format(_sym.lower()), title='{} OHLCV and coinmetrics.io data'.format(_sym), figsize=(40,30))
    corr.to_excel('data/datasets/resampled_ohlcv_ta/correlation/{}.xlsx'.format(_sym.lower()), index=True, index_label='Date')
    index[_sym] = {'csv':csv_path, 'xls':xls_path}

    print('Saved {} in data/datasets/resampled_ohlcv_ta/'.format(_sym))

with open('data/datasets/resampled_ohlcv_ta/index.json', 'w') as f:
    json.dump(index, f, sort_keys=True, indent=4)