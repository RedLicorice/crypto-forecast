import logging
from lib.log import logger
from lib.symbol import Symbol, DatasetType
from lib.models import ModelFactory
from lib.job import Job
from lib.report import ReportCollection
import pandas as pd

logger.setup(
    filename='../job_test.log',
    filemode='w',
    root_level=logging.DEBUG,
    log_level=logging.DEBUG,
    logger='job_test'
)
ModelFactory.discover()

ohlcv = pd.read_csv("./data/result/ohlcv.csv", sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
btc = pd.read_csv("./data/coinmetrics.io/btc.csv", sep=',', encoding='utf-8', index_col='date', parse_dates=True)

_sym = 'BTC'
s = Symbol(_sym, ohlcv=ohlcv, blockchain=btc, column_map={
    'open': _sym+'_Open',
    'high': _sym+'_High',
    'low': _sym+'_Low',
    'close': _sym,
    'volume': _sym+'_Volume'
})

# bchn = s.get_dataset(DatasetType.BLOCKCHAIN)
# correlation(bchn.corr(), 'data/result/blockchain-corr.png', figsize=(32,18))
# pat = s.get_dataset(DatasetType.OHLCV_PATTERN)
# bchn.to_csv('data/result/block1chain-dataset.csv', sep=',', encoding='utf-8', index=True, index_label='Date')
# pat.to_csv('data/result/ohlcv_pattern.csv', sep=',', encoding='utf-8', index=True, index_label='Date')
# fourier_transform(bchn['CapMVRVCur'])
m = ModelFactory.create_model('mlp')
#s = s.add_lag(7)
s = s.time_slice('2018-01-01', '2018-02-27', format='%Y-%m-%d')
j = Job(symbol=s, model=m)
reports = j.grid_search(x_type=DatasetType.CONTINUOUS_TA,
                    y_type=DatasetType.DISCRETE_TA,
                    #undersample=True,
                    #multiprocessing=False,
                    #discretize=False,
                    #variance_threshold=0.01,
                )

# Common
if isinstance(reports, list):
    c = ReportCollection(reports)
    df = c.to_dataframe()
    print(df.head())
    br = min(reports)
    print('Best config:\n\t{} accuracy: {} mse: {} profit: {}%'.format(str(br), str(br.accuracy()), str(br.mse()), br.profit()))
else:
    print('{} accuracy: {} mse: {} profit: {}%'.format(str(reports), str(reports.accuracy()), str(reports.mse()), reports.profit()))
    br = reports

#signal_plot(ohlcv, result)
br.plot_signals()
