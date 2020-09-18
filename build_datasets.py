import logging
from lib.log import logger
import pandas as pd
import numpy as np
from lib.dataset import *
from lib.plot import decompose_feature
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import json, os


def decompose_dataframe_features(dataset, _sym, _df):
    # Create feature plots
    for c in _df.columns:
        years = [g for n, g in _df[c].groupby(pd.Grouper(freq='Y'))] # Split dataset by year
        for y in years:
            year = y.index.year[0]
            decompose_feature("{}.{} ({}) {}".format(dataset, _sym, year, c), y)
            os.makedirs('data/datasets/{}/decomposed/{}/{}/'.format(dataset, _sym, c), exist_ok=True)
            plt.savefig('data/datasets/{}/decomposed/{}/{}/{}.png'.format(dataset, _sym, c, year))
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
            logger.info('Plotting features {}'.format(_sym))
            for c in _df.columns:
                fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 16))
                axes[0].set_title(label='{} {} values'.format(_sym, c))
                axes[1].set_title(label='{} {} distribution'.format(_sym, c))
                _df[c].plot(ax=axes[0])
                _df[c].hist(ax=axes[1])
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

def build_old_dataset():
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

def build_merged_dataset():
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

def build_atsa_dataset(source_index, W=10):
    _dataset = load_dataset(source_index, return_index=True)
    index = {}

    for _sym, entry in _dataset.items():
        _df = pd.read_csv(entry['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        _target = pd.read_csv(entry['target_csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        ohlcv = _df[entry['features']['ohlcv']]
        ta = _df[entry['features']['ta']]

        # Build the dataframe with base features
        ohlc = ohlcv[['open','high','low','close']]
        lagged_ohlc = pd.concat([ohlc] + [builder.make_lagged(ohlc, i) for i in range(1, W + 1)], axis='columns', verify_integrity=True, sort=True, join='inner')
        # Add lagged features to the dataframe
        atsa_df = pd.concat([lagged_ohlc, ta], axis='columns', verify_integrity=True, sort=True, join='inner')

        # Drop the first 30 rows
        atsa_df = atsa_df[30:]

        # Save the dataframe
        atsa_df.to_csv('data/datasets/all_merged/csv/{}_atsa.csv'.format(_sym.lower()), sep=',',
                           encoding='utf-8', index=True,
                           index_label='Date')
        atsa_df.to_excel('data/datasets/all_merged/excel/{}_atsa.xlsx'.format(_sym.lower()),
                             index=True, index_label='Date')
        # decompose_dataframe_features('all_merged', _sym+'_improved', unlagged_df)
        # Add symbol to index
        index[_sym] = {
            'csv': 'data/datasets/all_merged/csv/{}_atsa.csv'.format(_sym.lower()),
            'xls': 'data/datasets/all_merged/excel/{}_atsa.xlsx'.format(_sym.lower()),
            'target_csv': 'data/datasets/all_merged/csv/{}_target.csv'.format(_sym.lower()),
            'target_xls': 'data/datasets/all_merged/excel/{}_target.xlsx'.format(_sym.lower()),
            'features': {
                'atsa': [c for c in atsa_df.columns],
            }
        }
        logger.info('Saved {} in data/datasets/all_merged/'.format(_sym))
    with open('data/datasets/all_merged/index_atsa.json', 'w') as f:
        json.dump(index, f, sort_keys=True, indent=4)

def build_improved_dataset(source_index):
    _dataset = load_dataset(source_index, return_index=True)
    index = {}

    for _sym, entry in _dataset.items():
        _df = pd.read_csv(entry['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        _target = pd.read_csv(entry['target_csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        ohlcv = _df[entry['features']['ohlcv']]

        ta = _df[entry['features']['ta']]
        ta_7 = _df[entry['features']['ta_7d']]
        cm = _df[entry['features']['cm']]

        ohlcv_stats = pd.DataFrame(index=ohlcv.index)
        ohlcv_stats['volume'] = ohlcv.volume
        ohlcv_stats['volume_pct'] = ohlcv.volume.pct_change()
        ohlcv_stats['close_pct'] = ohlcv.close.pct_change()
        ohlcv_stats['day_range'] = ohlcv.high - ohlcv.low # Showld always be > 0, price oscillation range for current day
        ohlcv_stats['direction'] = ohlcv.close - ohlcv.open # Price direction for the day green > 0, red < 0. Modulus is range.

        cm_picked = pd.DataFrame(index=ohlcv.index)
        if'adractcnt' in cm.columns:
            cm_picked['adractcnt_pct'] = cm.adractcnt.pct_change()
            cm_picked['adractcnt_mean3_pct'] = cm.adractcnt.rolling(3).mean().pct_change()
        if 'splycur' in cm.columns:
            cm_picked['vol_supply'] = ohlcv.volume / cm.splycur # Ratio between transacted volume and total supply (mined)
        if 'txtfrvaladjntv' in cm.columns and 'isstotntv' in cm.columns:
            cm_picked['tx_issued'] = cm.txtfrvaladjntv / cm.isstotntv # Ratio between transacted coins and issued coins that day
        if 'hashrate' in cm.columns:
            cm_picked['hashrate_mean3_pct'] = cm.hashrate.rolling(3).mean().pct_change()
        if 'roi30d' in cm.columns:
            cm_picked['roi30d'] = cm.roi30d
        if 'isstotntv' in cm.columns:
            cm_picked['isstotntv_pct'] = cm.isstotntv.pct_change()
        if 'feetotntv' in cm.columns:
            cm_picked['feetotntv_pct'] = cm.feetotntv.pct_change()
        if 'txtfrcount' in cm.columns:
            cm_picked['txtfrcount_pct'] = cm.txtfrcount.pct_change()
            cm_picked['txtfrcount_volume'] = cm.txtfrcount.pct_change()
        if 'vtydayret30d' in cm.columns:
            cm_picked['vtydayret30d'] = cm.vtydayret30d
        if 'isscontpctann' in cm.columns:
            cm_picked['isscontpctann'] = cm.isscontpctann

        ta_picked = pd.DataFrame(index=ta.index)
        ta_picked['rema_5_20'] = ta.rema_5_20
        ta_picked['rema_8_15'] = ta.rema_8_15
        ta_picked['rema_20_50'] = ta.rema_20_50
        ta_picked['rsma_5_20'] = ta.rema_5_20
        ta_picked['rsma_8_15'] = ta.rema_8_15
        ta_picked['rsma_20_50'] = ta.rema_20_50
        ta_picked['stoch_14'] = ta.stoch_14
        ta_picked['pvo_12_26'] = ta.pvo_12_26
        ta_picked['macd_12_26'] = ta.macd_12_26
        ta_picked['adi_pct'] = ta.adi.pct_change()
        ta_picked['rsi_14_pct'] = ta.rsi_14.pct_change()

        #lagged_stats = pd.concat([ohlcv_stats] + [builder.make_lagged(ohlcv_stats, i) for i in range(1,10+1)], axis='columns', verify_integrity=True, sort=True, join='inner')

        # Build the dataframe with base features
        lagged_close = pd.concat([ohlcv.close] + [builder.make_lagged(ohlcv.close, i) for i in range(1,10+1)], axis='columns', verify_integrity=True, sort=True, join='inner')
        lagged_close.columns = ['close'] + ['close_lag-{}'.format(i) for i in range(1,10+1)]
        # Add lagged features to the dataframe
        improved_df = pd.concat([ohlcv_stats, lagged_close, cm_picked, ta_picked], axis='columns', verify_integrity=True, sort=True, join='inner')

        # Drop the first 30 rows
        improved_df = improved_df[30:]

        # Save the dataframe
        improved_df.to_csv('data/datasets/all_merged/csv/{}_improved.csv'.format(_sym.lower()), sep=',', encoding='utf-8', index=True,
                  index_label='Date')
        improved_df.to_excel('data/datasets/all_merged/excel/{}_improved.xlsx'.format(_sym.lower()), index=True, index_label='Date')
        unlagged_df = improved_df[[c for c in improved_df.columns if not '_lag' in c]]
        #decompose_dataframe_features('all_merged', _sym+'_improved', unlagged_df)
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
    # Find common features
    common_features = []
    for _sym, entry in index.items():
        features = entry['features']['improved']
        if not common_features: # if common_features is empty, common_features are all the current features
            common_features = features
        not_common_features = []
        for f in common_features: # remove features from common_features which are not in features
            if f not in features:
                not_common_features.append(f)
        for f in not_common_features:
            common_features.remove(f)
    for _sym, entry in index.items():
        entry['features']['common'] = common_features
    # Save index again
    with open('data/datasets/all_merged/index_improved.json', 'w') as f:
        json.dump(index, f, sort_keys=True, indent=4)



if __name__ == '__main__':
    logger.setup(
        filename='../build_dataset.log',
        filemode='w',
        root_level=logging.DEBUG,
        log_level=logging.DEBUG,
        logger='build_dataset'
    )
    build_merged_dataset()
    build_atsa_dataset('all_merged')
    build_improved_dataset('all_merged')