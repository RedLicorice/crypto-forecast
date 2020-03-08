import logging
from lib.log import logger
from lib.symbol import Symbol, DatasetType
from lib.job import Job
import os
import pandas as pd
from lib.plotter import correlation, save_plot
from lib.models import ModelFactory
from lib.report import ReportCollection

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
ModelFactory.discover()

ohlcv = pd.read_csv("./data/result/ohlcv.csv", sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
chain = pd.read_csv("./data/result/blockchains.csv", sep=',', encoding='utf-8', index_col='Date', parse_dates=True)

models = ModelFactory.create_all(['arima','expsmooth','sarima', 'nn', 'kmeans'])
for _sym in SYMBOLS:
    s = Symbol(_sym, ohlcv=ohlcv, blockchain=chain[[c for c in chain.columns if c.startswith(_sym)]], column_map={
        'open': _sym+'_Open',
        'high': _sym+'_High',
        'low': _sym+'_Low',
        'close': _sym,
        'volume': _sym+'_Volume'
    }).time_slice('2018-01-01', '2018-02-27', format='%Y-%m-%d')
    jobs = [Job(symbol=s, model=m) for m in models]
    report_groups = [j.grid_search(x_type=DatasetType.CONTINUOUS_TA, y_type=DatasetType.DISCRETE_TA) for j in jobs]

    for reports in report_groups:
        c = ReportCollection(reports)
        df = c.to_dataframe()
        br = min(reports)
        print('Best config:\n\t{} accuracy: {} mse: {} profit: {}%'.format(str(br), str(br.accuracy()), str(br.mse()), br.profit()))
        _name = 'gridsearch-{}-{}'.format(br.model, str(br.symbol))
        br.plot_signals(save_to='data/reports/{}-best_mse.png'.format(_name))
        df.to_csv('data/reports/{}.csv'.format(_name), sep=',', encoding='utf-8', index=True, index_label='Date')


