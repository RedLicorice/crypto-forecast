import logging
from lib.log import logger
from lib.symbol import Symbol, DatasetType
from lib.models.arima import ARIMAModel
from lib.job import Job
import pandas as pd

logger.setup(
    filename='../job_test.log',
    filemode='w',
    root_level=logging.DEBUG,
    log_level=logging.DEBUG,
    logger='job_test'
)
df = pd.read_csv("./data/result/ohlcv.csv", sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
btc = pd.read_csv("./data/coinmetrics.io/btc.csv", sep=',', encoding='utf-8', index_col='date', parse_dates=True)
s = Symbol('BTC', ohlcv=df, blockchain=btc, column_map={
    'open': 'BTC_Open',
    'high': 'BTC_High',
    'low': 'BTC_Low',
    'close': 'BTC',
    'volume': 'BTC_Volume'
})
m = ARIMAModel()
s = s.time_slice('2016-12-01', '2016-12-31', format='%Y-%m-%d')
j = Job(symbol=s, model=m)
r = j.holdout(x_type=DatasetType.OHLCV_PCT, y_type=DatasetType.OHLCV_PCT, univariate_column='close')

#r = min(reports)
print('Best config: {} mse: {}'.format(str(r), str(r.mse())))