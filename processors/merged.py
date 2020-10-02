from lib.log import logger
from lib.dataset import builder, load_preprocessed, save_symbol_dataset
import pandas as pd

NEED_SOURCE_INDEX=False

def build(source_index, dest_index, W=10):
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
        close = ohlcv['close']
        target_pct = builder.target_price_variation(ohlcv['close'], periods=1)
        target_class = builder.target_discrete_price_variation(target_pct)
        target_binary = builder.target_binary_price_variation(target_pct)
        target_labels = builder.target_label(target_class, labels=['SELL', 'HOLD', 'BUY'])
        target_binary_labels = builder.target_label(target_binary, labels=['SELL', 'BUY'])
        target_bin = builder.target_binned_price_variation(target_pct, n_bins=3)
        target_bin_binary = builder.target_binned_price_variation(target_pct, n_bins=2)
        target_bin_labels = builder.target_label(target_bin, labels=['SELL', 'HOLD', 'BUY'])
        target_bin_binary_labels = builder.target_label(target_bin_binary, labels=['SELL','BUY'])
        # Merge all the datasets
        dataframes = [ohlcv, ohlcv_3d, ohlcv_7d, ohlcv_30d, ta, ta_3d, ta_7d, ta_30d, cm]#, social_pct]
        df = pd.concat(dataframes, axis='columns', verify_integrity=True, sort=True, join='inner')
        target = pd.concat([close, target_pct, target_class, target_binary, target_bin, target_bin_binary, target_labels, target_binary_labels, target_bin_labels, target_bin_binary_labels], axis=1)
        target.columns = [
            'close',
            'pct',
            'class',
            'binary',
            'bin',
            'binary_bin',
            'labels',
            'binary_labels',
            'bin_labels',
            'binary_bin_labels'
        ]
        target = target.loc[df.first_valid_index():df.last_valid_index()]
        # Save resulting dataset both in CSV and Excel format
        logger.info('Saving {}'.format(_sym))
        feature_groups = {
            'ohlcv': [c for c in ohlcv.columns],
            'ohlcv_3d': [c for c in ohlcv_3d.columns],
            'ohlcv_7d': [c for c in ohlcv_7d.columns],
            'ohlcv_30d': [c for c in ohlcv_30d.columns],
            'ta': [c for c in ta.columns],
            'ta_3d': [c for c in ta_3d.columns],
            'ta_7d': [c for c in ta_7d.columns],
            'ta_30d': [c for c in ta_30d.columns],
            'cm': [c for c in cm.columns],
            # 'social_pct': [c for c in social_pct.columns],
        }
        save_symbol_dataset(dest_index, _sym, df, target=target, feature_groups=feature_groups)
        logger.info('Saved {}'.format(_sym))