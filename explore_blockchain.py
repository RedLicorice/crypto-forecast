import logging
from lib.log import logger
from lib.symbol import Symbol, DatasetType
from lib.models.mlp import MLPModel
from lib.job import Job
from lib.plotter import lineplot, scatter, correlation, fourier_transform, signal_plot
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

#bchn = s.get_dataset(DatasetType.BLOCKCHAIN)
# correlation(bchn.corr(), 'data/result/blockchain-corr.png', figsize=(32,18))
# pat = s.get_dataset(DatasetType.OHLCV_PATTERN)
# bchn.to_csv('data/result/block1chain-dataset.csv', sep=',', encoding='utf-8', index=True, index_label='Date')
# pat.to_csv('data/result/ohlcv_pattern.csv', sep=',', encoding='utf-8', index=True, index_label='Date')
# fourier_transform(bchn['CapMVRVCur'])
m = MLPModel()
s = s.time_slice('2018-01-01', '2018-05-30', format='%Y-%m-%d')
j = Job(symbol=s, model=m)
reports = j.grid_search(dataset=DatasetType.CONTINUOUS_TA,
                        target=DatasetType.DISCRETE_TA,
                        multiprocessing=True,
                        # oversample=True,
                        # undersample=True,
                        params={
                            'activation': 'tanh',
                            'solver': 'adam',
                            'hidden_layer_sizes': (10, 4),
                            'learning_rate_init': 0.01,
                            'learning_rate': 'invscaling'
                        })

# Common
if isinstance(reports, list):
    for r in reports:
         print('Evaluation: {} accuracy: {} mse: {}'.format(str(r), str(r.accuracy()), str(r.mse())))
    br = min(reports)
    #scatter(result.index, [result['prediction'].values, result['result'].values])
    print('Best config: {} accuracy: {}'.format(str(br), str(br.accuracy())))
else:
    print('{} accuracy: {} profit: {}%'.format(str(reports), str(reports.accuracy()), reports.profit()))
    br = reports

#signal_plot(ohlcv, result)
br.plot_signals()
