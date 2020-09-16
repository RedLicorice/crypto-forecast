import logging
from lib.log import logger
import pandas as pd
import numpy as np
from lib.dataset import *
from lib.plot import decompose_feature
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, KBinsDiscretizer
from sklearn.impute import SimpleImputer
import json, os


def build_dataset():
    ohlcv_index = load_preprocessed('ohlcv')
    cm_index = load_preprocessed('coinmetrics.io')
    #social_index = load_preprocessed('cryptocompare_social')
    index = {}
    for _sym in ohlcv_index.keys():
        if not _sym in cm_index:
            logger.warning('Missing blockchain data for {}'.format(_sym))
            continue
        # if not _sym in social_index:
        #     logger.warning('Missing social data for {}'.format(_sym))
        #     continue
        logger.info('Building {}'.format(_sym))
        ohlcv = pd.read_csv(ohlcv_index[_sym]['csv'], sep=',', encoding='utf-8',
                        index_col='Date', parse_dates=True)
        cm = pd.read_csv(cm_index[_sym]['csv'], sep=',', encoding='utf-8',
                        index_col='Date', parse_dates=True)
        #social = pd.read_csv(social_index[_sym]['csv'], sep=',', encoding='utf-8',
        #                index_col='Date', parse_dates=True)
        # Build resampled OHLCV and TA features
        ohlcv_3d = builder.periodic_ohlcv_pct_change(ohlcv, period=3, label=True)
        ohlcv_7d = builder.periodic_ohlcv_pct_change(ohlcv, period=7, label=True)
        ohlcv_30d = builder.periodic_ohlcv_pct_change(ohlcv, period=30, label=True)
        ta = builder.features_ta(ohlcv)
        ta_3d = builder.period_resampled_ta(ohlcv, period=3)
        ta_7d = builder.period_resampled_ta(ohlcv, period=7)
        ta_30d = builder.period_resampled_ta(ohlcv, period=30)
        # Build Coinmetrics blockchain stats
        cm_pct = feature_quality_filter(builder.pct_change(cm))
        # Build Cryptocompare social stats
        #social_pct = feature_quality_filter(builder.pct_change(social))
        # Build target percent variation
        target_pct = builder.target_price_variation(ohlcv['close'], periods=1)
        target_class = builder.target_discrete_price_variation(target_pct)
        target_labels = builder.target_label(target_class, labels=['SELL', 'HOLD', 'BUY'])
        target_bin = builder.target_binned_price_variation(target_pct, n_bins=3)
        target_bin_binary = builder.target_binned_price_variation(target_pct, n_bins=2)
        target_bin_labels = builder.target_label(target_bin, labels=['SELL', 'HOLD', 'BUY'])
        target_bin_binary_labels = builder.target_label(target_bin_binary, labels=['SELL','BUY'])
        # Merge all the datasets
        dataframes = [ohlcv, ohlcv_3d, ohlcv_7d, ohlcv_30d, ta, ta_3d, ta_7d, ta_30d, cm_pct]#, social_pct]
        df = pd.concat(dataframes, axis='columns', verify_integrity=True, sort=True, join='inner')
        target = pd.concat([target_pct, target_class, target_bin, target_bin_binary, target_labels, target_bin_labels, target_bin_binary_labels], axis=1)
        target.columns = [
            'pct',
            'class',
            'bin',
            'binary_bin',
            'labels',
            'bin_labels',
            'binary_bin_labels'
        ]
        target = target.loc[df.first_valid_index():df.last_valid_index()]
        # Save resulting dataset both in CSV and Excel format
        logger.info('Saving {}'.format(_sym))

        df.to_csv('data/datasets/all_merged/csv/{}.csv'.format(_sym.lower()), sep=',', encoding='utf-8', index=True, index_label='Date')
        df.to_excel('data/datasets/all_merged/excel/{}.xlsx'.format(_sym.lower()), index=True, index_label='Date')
        target.to_csv('data/datasets/all_merged/csv/{}_target.csv'.format(_sym.lower()), sep=',', encoding='utf-8', index=True, index_label='Date')
        target.to_excel('data/datasets/all_merged/excel/{}_target.xlsx'.format(_sym.lower()), index=True, index_label='Date')

        # Add symbol to index
        index[_sym] = {
            'csv': 'data/datasets/all_merged/csv/{}.csv'.format(_sym.lower()),
            'xls': 'data/datasets/all_merged/excel/{}.xlsx'.format(_sym.lower()),
            'target_csv': 'data/datasets/all_merged/csv/{}_target.csv'.format(_sym.lower()),
            'target_xls': 'data/datasets/all_merged/excel/{}_target.xlsx'.format(_sym.lower()),
            'features': {
                'ohlcv': [c for c in ohlcv.columns],
                'ohlcv_3d': [c for c in ohlcv_3d.columns],
                'ohlcv_7d': [c for c in ohlcv_7d.columns],
                'ohlcv_30d': [c for c in ohlcv_30d.columns],
                'ta': [c for c in ta.columns],
                'ta_3d': [c for c in ta_3d.columns],
                'ta_7d': [c for c in ta_7d.columns],
                'ta_30d': [c for c in ta_30d.columns],
                'cm_pct': [c for c in cm_pct.columns],
                #'social_pct': [c for c in social_pct.columns],
            }
        }

        logger.info('Saved {} in data/datasets/all_merged/'.format(_sym))
    with open('data/datasets/all_merged/index.json', 'w') as f:
        json.dump(index, f, sort_keys=True, indent=4)

def build_simple_dataset():
    ohlcv_index = load_preprocessed('ohlcv')
    cm_index = load_preprocessed('coinmetrics.io')
    index = {}

    for _sym in ohlcv_index.keys():
        if not _sym in cm_index:
            logger.warning('Missing blockchain data for {}'.format(_sym))
            continue

        logger.info('Building {}'.format(_sym))
        ohlcv = pd.read_csv(ohlcv_index[_sym]['csv'], sep=',', encoding='utf-8',
                        index_col='Date', parse_dates=True)
        cm = pd.read_csv(cm_index[_sym]['csv'], sep=',', encoding='utf-8',
                        index_col='Date', parse_dates=True)

        # Build resampled OHLCV and TA features
        ohlcv_3d = builder.periodic_ohlcv_resample(ohlcv, period=3, label=True)
        ohlcv_7d = builder.periodic_ohlcv_resample(ohlcv, period=7, label=True)
        ohlcv_30d = builder.periodic_ohlcv_resample(ohlcv, period=30, label=True)
        ta = builder.features_ta(ohlcv)
        ta_3d = builder.period_resampled_ta(ohlcv, period=3)
        ta_7d = builder.period_resampled_ta(ohlcv, period=7)
        ta_30d = builder.period_resampled_ta(ohlcv, period=30)

        # Build target percent variation
        target_pct = builder.target_price_variation(ohlcv['close'], periods=1)
        target_class = builder.target_discrete_price_variation(target_pct)
        target_labels = builder.target_label(target_class, labels=['SELL', 'HOLD', 'BUY'])
        target_bin = builder.target_binned_price_variation(target_pct, n_bins=3)
        target_bin_binary = builder.target_binned_price_variation(target_pct, n_bins=2)
        target_bin_labels = builder.target_label(target_bin, labels=['SELL', 'HOLD', 'BUY'])
        target_bin_binary_labels = builder.target_label(target_bin_binary, labels=['SELL','BUY'])
        # Merge all the datasets
        dataframes = [ohlcv, ohlcv_3d, ohlcv_7d, ohlcv_30d, ta, ta_3d, ta_7d, ta_30d, cm]#, social_pct]
        df = pd.concat(dataframes, axis='columns', verify_integrity=True, sort=True, join='inner')
        target = pd.concat([target_pct, target_class, target_bin, target_bin_binary, target_labels, target_bin_labels, target_bin_binary_labels], axis=1)
        target.columns = [
            'pct',
            'class',
            'bin',
            'binary_bin',
            'labels',
            'bin_labels',
            'binary_bin_labels'
        ]
        target = target.loc[df.first_valid_index():df.last_valid_index()]
        # Save resulting dataset both in CSV and Excel format
        logger.info('Saving {}'.format(_sym))

        df.to_csv('data/datasets/all_merged/csv/{}.csv'.format(_sym.lower()), sep=',', encoding='utf-8', index=True, index_label='Date')
        df.to_excel('data/datasets/all_merged/excel/{}.xlsx'.format(_sym.lower()), index=True, index_label='Date')
        target.to_csv('data/datasets/all_merged/csv/{}_target.csv'.format(_sym.lower()), sep=',', encoding='utf-8', index=True, index_label='Date')
        target.to_excel('data/datasets/all_merged/excel/{}_target.xlsx'.format(_sym.lower()), index=True, index_label='Date')

        # Add symbol to index
        index[_sym] = {
            'csv': 'data/datasets/all_merged/csv/{}.csv'.format(_sym.lower()),
            'xls': 'data/datasets/all_merged/excel/{}.xlsx'.format(_sym.lower()),
            'target_csv': 'data/datasets/all_merged/csv/{}_target.csv'.format(_sym.lower()),
            'target_xls': 'data/datasets/all_merged/excel/{}_target.xlsx'.format(_sym.lower()),
            'features': {
                'ohlcv': [c for c in ohlcv.columns],
                'ohlcv_3d': [c for c in ohlcv_3d.columns],
                'ohlcv_7d': [c for c in ohlcv_7d.columns],
                'ohlcv_30d': [c for c in ohlcv_30d.columns],
                'ta': [c for c in ta.columns],
                'ta_3d': [c for c in ta_3d.columns],
                'ta_7d': [c for c in ta_7d.columns],
                'ta_30d': [c for c in ta_30d.columns],
                'cm': [c for c in cm.columns],
                #'social_pct': [c for c in social_pct.columns],
            }
        }

        logger.info('Saved {} in data/datasets/all_merged/'.format(_sym))
    with open('data/datasets/all_merged/index.json', 'w') as f:
        json.dump(index, f, sort_keys=True, indent=4)



def improve_dataset_features(dataset):
    # Create feature plots
    _dataset = load_dataset(dataset, return_index=True)
    index = {}

    for _sym, entry in _dataset.items():
        _df = pd.read_csv(entry['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        ohlcv = _df[entry['features']['ohlcv']]
        ta = _df[entry['features']['ta']]
        cm = _df[entry['features']['cm']].copy()
        for c in cm.columns:
            series = cm[c].dropna()
            if series.shape[0] <= 0:
                continue
            if not is_stationary(series):
                cm[c] = cm[c].pct_change()

        improved_df = pd.concat([ohlcv.pct_change(), ta] + [builder.make_lagged(ohlcv.pct_change(), i) for i in range(1,7+1)], axis='columns', verify_integrity=True, sort=True, join='inner')
        improved_df['day_range'] = ohlcv.close - ohlcv.open # direction of today's price action
        improved_df.to_csv('data/datasets/all_merged/csv/{}_improved.csv'.format(_sym.lower()), sep=',', encoding='utf-8', index=True,
                  index_label='Date')
        improved_df.to_excel('data/datasets/all_merged/excel/{}_improved.xlsx'.format(_sym.lower()), index=True, index_label='Date')
        # Add symbol to index
        index[_sym] = {
            'csv': 'data/datasets/all_merged/csv/{}_improved.csv'.format(_sym.lower()),
            'xls': 'data/datasets/all_merged/excel/{}_improved.xlsx'.format(_sym.lower()),
            'target_csv': 'data/datasets/all_merged/csv/{}_target.csv'.format(_sym.lower()),
            'target_xls': 'data/datasets/all_merged/excel/{}_target.xlsx'.format(_sym.lower()),
            'features': {
                'improved': [c for c in improved_df.columns],
            }
        }
        logger.info('Saved {} in data/datasets/all_merged/'.format(_sym))
    with open('data/datasets/all_merged/index_improved.json', 'w') as f:
        json.dump(index, f, sort_keys=True, indent=4)



def decompose_dataset_features(dataset):
    # Create feature plots
    _dataset = load_dataset(dataset)
    for _sym, (_df, _target) in _dataset.items():
        for c in _df.columns:
            years = [g for n, g in _df[c].groupby(pd.Grouper(freq='Y'))] # Split dataset by year
            for y in years:
                year = y.index.year[0]
                decompose_feature("{}.{} ({}) {}".format(dataset, _sym, year, c), y)
                os.makedirs('data/datasets/all_merged/decomposed/{}/{}/'.format(_sym, c), exist_ok=True)
                plt.savefig('data/datasets/all_merged/decomposed/{}/{}/{}.png'.format(_sym, c, year))
                plt.close()

def plot_dataset_features(dataset, plot_features=True, plot_target=False):
    # Create feature plots
    _dataset = load_dataset(dataset)
    for _sym, (_df, _target) in _dataset.items():
        logger.info('Plotting features for {}'.format(_sym))
        if plot_features:
            _df = _df.replace([np.inf, -np.inf], np.nan).dropna(how='all', axis=1)
            imputer = SimpleImputer(strategy='mean')
            values = imputer.fit_transform(_df.values)
            scaler = RobustScaler()
            values = scaler.fit_transform(values)
            _df = pd.DataFrame(values, index=_df.index, columns=_df.columns)
            discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
            discrete_values = discretizer.fit_transform(_df)
            _df_discrete = pd.DataFrame(discrete_values, index=_df.index, columns=_df.columns)
            logger.info('Plotting features {}'.format(_sym))
            for c in _df.columns:
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
                axes[0, 0].set_title(label='{} {} values'.format(_sym, c))
                axes[1, 0].set_title(label='{} {} distribution'.format(_sym, c))
                axes[0, 1].set_title(label='{} {} discrete values'.format(_sym, c))
                axes[1, 1].set_title(label='{} {} discrete distribution'.format(_sym, c))

                _df[c].plot(ax=axes[0, 0])
                _df[c].hist(ax=axes[1, 0])
                _df_discrete[c].plot(ax=axes[0, 1])
                _df_discrete[c].hist(ax=axes[1, 1])
                plt.savefig('data/datasets/all_merged/features/{}__{}.png'.format(_sym.lower(), c))
                plt.close()
        if plot_target:
            logger.info('Plotting target {}'.format(_sym))
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 16))
            axes[0, 0].set_title(label='{} target percent variation'.format(_sym))
            axes[1, 0].set_title(label='{} target percent variation distribution'.format(_sym))
            axes[0, 1].set_title(label='{} classes distribution'.format(_sym))
            axes[1, 1].set_title(label='{} bins distribution'.format(_sym))
            axes[0, 2].set_title(label='{} binary bins distribution'.format(_sym))

            _target = _target.replace([np.inf, -np.inf], np.nan)
            _target['pct'].fillna(method='ffill').plot(ax=axes[0, 0])
            _target['pct'].fillna(method='ffill').hist(ax=axes[1, 0])
            _target['class'].hist(ax=axes[0, 1])
            _target['bin'].hist(ax=axes[1, 1])
            _target['binary_bin'].hist(ax=axes[0, 2])
            plt.savefig('data/datasets/all_merged/targets/{}.png'.format(_sym.lower()))
            plt.close()

if __name__ == '__main__':
    logger.setup(
        filename='../build_dataset.log',
        filemode='w',
        root_level=logging.DEBUG,
        log_level=logging.DEBUG,
        logger='build_dataset'
    )
    #build_simple_dataset()
    improve_dataset_features('all_merged')