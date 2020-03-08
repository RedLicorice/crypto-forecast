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
    # 'ADA',
    # 'BCH',
    # 'BNB',
    'BTC',
    # 'BTG',
    # 'DASH',
    # 'DOGE',
    # 'EOS',
    # 'ETC',
    # 'ETH',
    # 'LTC',
    # 'LINK',
    # 'NEO',
    # 'QTUM',
    # 'TRX',
    # 'USDT',
    # 'VEN',
    # 'WAVES',
    # 'XEM',
    # 'XMR',
    # 'XRP',
    # 'ZEC',
    # 'ZRX'
]

logger.setup(
    filename='../job_test.log',
    filemode='w',
    root_level=logging.DEBUG,
    log_level=logging.DEBUG,
    logger='job_test'
)
ModelFactory.discover()

all_ohlcv = pd.read_csv("./data/result/ohlcv.csv", sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
all_chains = pd.read_csv("./data/result/blockchains.csv", sep=',', encoding='utf-8', index_col='Date', parse_dates=True)

IND_TA_7 = {
	'rsma' : [(5,20), (8,15), (20,50)],
	'rema' : [(5,20), (8,15), (20,50)],
	'macd' : [(12,26)],
	'ao' : [14],
	'adx' : [14],
	'wd' : [14],
	'ppo' : [(12,26)],
	'rsi':[14],
	'mfi':[14],
	'tsi':None,
	'stoch':[14],
	'cmo':[14],
	'atrp':[14],
	'pvo':[(12,26)],
	'fi':[13,50],
	'adi':None,
	'obv':None
}

models = ModelFactory.create_all(['arima','expsmooth','sarima', 'nn', 'kmeans'])
for _sym in SYMBOLS:
    ohlcv = builder.features_ohlcv(all_ohlcv, symbol=_sym)
    ohlcv_7 = builder.reverse_resample(ohlcv, period=7)
    ohlcv_7t = builder.features_timeframe(ohlcv, period=7)
    ohlcv_30 = builder.reverse_resample(ohlcv, period=30)
    ta_7d = builder.period_resampled_ta(ohlcv, period=7, indicators=IND_TA_7)


    if INTERACTIVE_FIGURE:
        fig = go.Figure(data=[
            # go.Ohlc(x=ohlcv_30d.index,
            #      open=ohlcv_30d['open'],
            #      high=ohlcv_30d['high'],
            #      low=ohlcv_30d['low'],
            #      close=ohlcv_30d['close'],
            #     increasing_line_color='cyan', decreasing_line_color='gray'
            # ),
            # go.Ohlc(x=ohlcv_7d.index,
            #      open=ohlcv_7d['open'],
            #      high=ohlcv_7d['high'],
            #      low=ohlcv_7d['low'],
            #      close=ohlcv_7d['close'],
            #      increasing_line_color = 'blue', decreasing_line_color = 'purple'
            # ),
            go.Ohlc(x=ohlcv.index,
                    open=ohlcv['open'],
                    high=ohlcv['high'],
                    low=ohlcv['low'],
                    close=ohlcv['close'],
                    increasing_line_color='lime', decreasing_line_color='red'
            ),
        ])
        fig.show()

    print("done")
