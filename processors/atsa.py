from lib.log import logger
from lib.dataset import builder, load_dataset, save_symbol_dataset
import pandas as pd


def build(source_index, dest_index, W=10):
    _dataset = load_dataset(source_index, return_index=True)

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
        logger.info('Saving {}'.format(_sym))
        save_symbol_dataset(dest_index, _sym, atsa_df)
        logger.info('Saved {}'.format(_sym))