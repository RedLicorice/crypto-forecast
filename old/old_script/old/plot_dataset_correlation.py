import logging
from lib.log import logger
import pandas as pd
from multiprocessing import freeze_support
from lib.plot import plot_correlation_matrix
import numpy as np
import json

def main():
    indexFile = 'data/datasets/ohlcv_coinmetrics/index.json'
    with open(indexFile) as f:
        index = json.load(f)

    for _sym, files in index.items():
        df = pd.read_csv(files['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # plot correlation matrix for differenced features
        df1 = df[[c for c in df.columns if c.endswith('_d1')] + ['open','high','low','close','volume']].copy()
        df1['target_pct'] = df['target_pct']
        df1['target'] = df['target']
        plot_correlation_matrix(df1.corr(), df1.columns, _sym + ' coinmetrics difference over 1 period', None,
                                'data/datasets/ohlcv_coinmetrics/correlation/' + _sym + '_d1.png', True)
        # plot correlation matrix for percent change features
        df2 = df[[c for c in df.columns if c.endswith('_p1')] + ['open','high','low','close','volume']].copy()
        df2['target_pct'] = df['target_pct']
        df2['target'] = df['target']
        plot_correlation_matrix(df2.corr(), df2.columns, _sym + ' coinmetrics percent change over 1 period', None,
                                'data/datasets/ohlcv_coinmetrics/correlation/' + _sym + '_p1.png', True)
        print(_sym)

if __name__ == '__main__':
    freeze_support()
    logger.setup(
        filename='../correlation.log',
        filemode='w',
        root_level=logging.DEBUG,
        log_level=logging.DEBUG,
        logger='correlation'
    )
    main()
