import logging
from lib.log import logger
from lib.symbol import Symbol, DatasetType
# Models
from lib.models.svc import SVCModel
from lib.models.mnb import MNBModel
from lib.models.mlp import MLPModel
from lib.models.neuralnet import NNModel
from lib.models.knn import KNNModel
from lib.models.expsmooth import ExpSmoothModel
from lib.models.arima import ARIMAModel
#
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

ohlcv = pd.read_csv("./data/result/ohlcv.csv", sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
blockchain = pd.read_csv("./data/result/blockchains.csv", sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
symbols = [x for x in ohlcv.columns if '_' not in x ]
_sym = 'ETH'
s = Symbol(_sym, ohlcv=ohlcv, blockchain=blockchain, column_map={
    'open': _sym+'_Open',
    'high': _sym+'_High',
    'low': _sym+'_Low',
    'close': _sym,
    'volume': _sym+'_Volume'
})

#bchn = s.get_dataset(DatasetType.BLOCKCHAIN)
# correlation(bchn.corr(), 'data/result/blockchain-corr.png', figsize=(32,18))
# pat = s.get_dataset(DatasetType.OHLCV_PATTERN)
# bchn.to_csv('data/result/block1chain-dataset.csv', sep=',', encoding='utf-8', index=True, index_label='Date')
# pat.to_csv('data/result/ohlcv_pattern.csv', sep=',', encoding='utf-8', index=True, index_label='Date')
# fourier_transform(bchn['CapMVRVCur'])
m = MNBModel()
s = s.time_slice('2018-01-01', '2018-02-27', format='%Y-%m-%d')
j = Job(symbol=s, model=m)
reports = j.grid_search(x_type=DatasetType.DISCRETE_TA,
                        y_type=DatasetType.DISCRETE_TA,
                        multiprocessing=False,
                        discretize=False,
                        variance_threshold=0.01,
                        params={'fit_prior': False, 'alpha': 0.01})

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
