from lib.log import logger
from lib.dataset import builder, load_dataset, save_symbol_dataset
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np


def build(source_index, dest_index, W=10):
    _dataset = load_dataset(source_index, return_index=True)
    for _sym, entry in _dataset.items():
        _df = pd.read_csv(entry['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        _target = pd.read_csv(entry['target_csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)

        ta = _df[entry['features']['ta']]
        cm = _df[entry['features']['cm']]

        # Price history facet (Daily variation of ohlc in last W trading days)
        ohlc = _df.loc[:,['open', 'high', 'low', 'close']]
        ohlc['open'] = STL(ohlc.open).fit().resid
        ohlc['high'] = STL(ohlc.high).fit().resid
        ohlc['low'] = STL(ohlc.low).fit().resid
        ohlc['close'] = STL(ohlc.close).fit().resid
        ohlc.columns = ['open_resid', 'high_resid', 'low_resid', 'close_resid']
        history_facet = pd.concat([ohlc] + [builder.make_lagged(ohlc, i) for i in range(1, W + 1)], axis='columns',
                                  verify_integrity=True, sort=True, join='inner')
        # Price trend facet (REMA/RSMA, MACD, AO, ADX, WD+ - WD-)
        trend_facet = ta[
            ["rsma_5_20", "rsma_8_15", "rsma_20_50", "rema_5_20", "rema_8_15", "rema_20_50", "macd_12_26", "ao_14",
             "adx_14", "wd_14"]]
        # Volatility facet (CMO, ATRp)
        volatility_facet = ta[["cmo_14", "atrp_14"]]
        # Volume facet (Volume pct, PVO, ADI, OBV)
        volume_facet = pd.concat(
            [_df.volume.pct_change().replace([np.inf, -np.inf], 0), ta[["pvo_12_26", "adi", "obv"]]], axis='columns',
            verify_integrity=True, sort=True, join='inner')
        # On-chain facet
        cm_1 = cm.reindex(columns=['adractcnt', 'txtfrvaladjntv', 'isstotntv', 'feetotntv', 'splycur', 'hashrate',
                                   'difficulty', 'txtfrcount']).pct_change()
        cm_2 = cm.reindex(columns=['isscontpctann'])
        chain_facet = pd.concat([cm_1, cm_2], axis='columns', verify_integrity=True, sort=True, join='inner')

        # Drop columns whose values are all nan or inf from each facet
        with pd.option_context('mode.use_inf_as_na', True):  # Set option temporarily
            history_facet = history_facet.dropna(axis='columns', how='all')
            trend_facet = trend_facet.dropna(axis='columns', how='all')
            volatility_facet = volatility_facet.dropna(axis='columns', how='all')
            volume_facet = volume_facet.dropna(axis='columns', how='all')
            chain_facet = chain_facet.dropna(axis='columns', how='all')

        improved_df = pd.concat([history_facet, trend_facet, volatility_facet, volume_facet, chain_facet],
                                axis='columns', verify_integrity=True, sort=True, join='inner')
        # Drop the first 30 rows
        #improved_df = improved_df[30:]

       # Add symbol to index
        feature_groups = {
            'price_history': [c for c in history_facet.columns],
            'trend': [c for c in trend_facet.columns],
            'volatility': [c for c in volatility_facet.columns],
            'volume': [c for c in volume_facet.columns],
            'chain': [c for c in chain_facet.columns],
        }
        logger.info('Saving {}'.format(_sym))
        save_symbol_dataset(dest_index, _sym, improved_df, feature_groups=feature_groups, target=_target)
        logger.info('Saved {}'.format(_sym))