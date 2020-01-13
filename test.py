import logging
from lib.log import logger
from lib.symbol import Symbol
from lib.models import ARIMAModel
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
s = Symbol('BTC', ohlcv=df, column_map={
    'open': 'BTC_Open',
    'high': 'BTC_High',
    'low': 'BTC_Low',
    'close': 'BTC',
    'volume': 'BTC_Volume'
})

m = ARIMAModel(params={'disp': True})
j = Job(symbol=s, model=m)
reports = j.grid_search(range=('2016-12-14', '2016-12-31'), multiprocessing=True)
r = min(reports)
print('Best config: {} mse: {}'.format(str(r), str(r.mse())))