import logging
from lib.log import logger
from lib.symbol import Symbol, DatasetType
from lib.job import Job
import os
import pandas as pd
from lib.plotter import correlation, save_plot
from lib.models import ModelFactory
import lib.dataset as builder
from lib.report import ReportCollection
import plotly.graph_objects as go

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

all_ohlcv = pd.read_csv("./data/result/ohlcv.csv", sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
all_chains = pd.read_csv("./data/result/blockchains.csv", sep=',', encoding='utf-8', index_col='Date', parse_dates=True)

for _sym in SYMBOLS:
    ohlcv = builder.load_ohlcv(all_ohlcv, symbol=_sym)
    ohlcv_7d = builder.periodic_ohlcv_pct_change(ohlcv, period=7, label=True)
    ohlcv_30d = builder.periodic_ohlcv_pct_change(ohlcv, period=30, label=True)
    ta = builder.features_ta(ohlcv)
    ta_7d = builder.period_resampled_ta(ohlcv, period=7)
    ta_30d = builder.period_resampled_ta(ohlcv, period=30)
    df = pd.concat([ohlcv, ohlcv_7d, ohlcv_30d, ta, ta_7d, ta_30d], axis='columns', verify_integrity=True, sort=True)
    df['target'] = builder.target_discrete_price_variation(ohlcv, periods=1)
    df['target_pct'] = builder.target_price_variation(ohlcv, periods=1)
    df['target_label'] = builder.target_discrete_price_variation(ohlcv, periods=1, labels=True)

    if INTERACTIVE_FIGURE:
        fig = go.Figure(data=[
            go.Ohlc(x=ohlcv_30d.index,
                 open=ohlcv_30d['open_30'],
                 high=ohlcv_30d['high_30'],
                 low=ohlcv_30d['low_30'],
                 close=ohlcv_30d['close_30'],
                increasing_line_color='cyan', decreasing_line_color='gray'
            ),
            go.Ohlc(x=ohlcv_7d.index,
                 open=ohlcv_7d['open_7'],
                 high=ohlcv_7d['high_7'],
                 low=ohlcv_7d['low_7'],
                 close=ohlcv_7d['close_7'],
                 increasing_line_color = 'blue', decreasing_line_color = 'purple'
            ),
            go.Ohlc(x=ohlcv.index,
                    open=ohlcv['open'],
                    high=ohlcv['high'],
                    low=ohlcv['low'],
                    close=ohlcv['close'],
                    increasing_line_color='lime', decreasing_line_color='red'
            ),
        ])
        fig.show()

    df.to_csv('data/result/datasets/{}_ohlcv_7_30__ta_7_30__target.csv'.format(_sym.lower()), sep=',', encoding='utf-8', index=True, index_label='Date')
    df.to_excel('data/result/datasets/{}_ohlcv_7_30__ta_7_30__target.xlsx'.format(_sym.lower()), index=True, index_label='Date')
    correlation(df.corr(), 'data/result/datasets/{}_ohlcv_7_30__ta_7_30__corr.png'.format(_sym.lower()), figsize=(64,48))
    print("{} done.".format(_sym))
