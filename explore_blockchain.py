import logging
from lib.log import logger
from lib.symbol import Symbol, DatasetType
from lib.models import SVCModel
from lib.job import Job
import pandas as pd

logger.setup(
    filename='../job_test.log',
    filemode='w',
    root_level=logging.DEBUG,
    log_level=logging.DEBUG,
    logger='job_test'
)

ohlcv = pd.read_csv("./data/result/ohlcv.csv", sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
btc = pd.read_csv("./data/coinmetrics.io/btc.csv", sep=',', encoding='utf-8', index_col='date', parse_dates=True)

s = Symbol('BTC', ohlcv=ohlcv, blockchain=btc, column_map={
    'open': 'BTC_Open',
    'high': 'BTC_High',
    'low': 'BTC_Low',
    'close': 'BTC',
    'volume': 'BTC_Volume'
})

# bchn = s.get_dataset(DatasetType.BLOCKCHAIN)
# pat = s.get_dataset(DatasetType.OHLCV_PATTERN)
# bchn.to_csv('data/result/blockchain-dataset.csv', sep=',', encoding='utf-8', index=True, index_label='Date')
# pat.to_csv('data/result/ohlcv_pattern.csv', sep=',', encoding='utf-8', index=True, index_label='Date')
#p.fourier_transform(bchn['CapMVRVCur'])
m = SVCModel()
s = s.time_slice('2016-12-01', '2016-12-31', format='%Y-%m-%d')
j = Job(symbol=s, model=m)
r = j.grid_search(dataset=DatasetType.OHLCV_PATTERN, multiprocessing=False)
#p.lineplot(bchn, [''])
r = min(r)
print('Best config: {} accuracy: {}'.format(str(r), str(r.accuracy())))