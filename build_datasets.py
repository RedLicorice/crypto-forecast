import logging
from lib.log import logger
from lib.plot import plot_correlation_matrix
import pandas as pd
import numpy as np
from lib.dataset import *
from lib.plot import decompose_feature
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy.stats import linregress
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

def convergence_between_series(s1: pd.Series, s2: pd.Series, W):
    # Fit a line on values in the window, then
    # return the angular coefficient's sign: 1 if positive, 0 otherwise
    def get_alpha(values: np.ndarray):
        slope, intercept, r_value, p_value, std_err = linregress(np.arange(0, values.size), values)
        #alpha = np.arctan(slope) / (np.pi / 2)
        return 1 if slope > 0 else 0
    # Map both series to their direction (0 decreasing, 1 increasing)
    s1_ = s1.rolling(W).apply(get_alpha, raw=True)
    s2_ = s2.rolling(W).apply(get_alpha, raw=True)
    # Result should be true if both values have the same direction (either both 0's or both 1's)
    #  XOR has the inverse of the truth table we need, so we just negate it.
    result = np.logical_not(np.logical_xor(s1_, s2_)).astype(int)
    return result

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

def build_improved_dataset(source_index, W=10):
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
        #ohlcv_stats['volume'] = ohlcv.volume
        #ohlcv_stats['volume_pct'] = ohlcv.volume.pct_change()
        #ohlcv_stats['close_pct'] = ohlcv.close.pct_change()
        ohlcv_stats['day_range_pct'] = (ohlcv.high - ohlcv.low).pct_change() # Showld always be > 0, price oscillation range for current day
        ohlcv_stats['direction'] = ohlcv.close - ohlcv.open # Price direction for the day green > 0, red < 0. Modulus is range.

        cm_picked = pd.DataFrame(index=ohlcv.index)
        if'adractcnt' in cm.columns:
            cm_picked['adractcnt_pct'] = cm.adractcnt.pct_change()
            # cm_picked['adractcnt_mean3_pct'] = cm.adractcnt.rolling(3).mean().pct_change()
            # cm_picked['adractcnt_mean7_pct'] = cm.adractcnt.rolling(7).mean().pct_change()
        # if 'splycur' in cm.columns: ## Correlated with volume and close
        #     cm_picked['vol_supply'] = ohlcv.volume / cm.splycur # Ratio between transacted volume and total supply (mined)
        if 'txtfrvaladjntv' in cm.columns and 'isstotntv' in cm.columns and 'feetotntv' in cm.columns:
            # I want to represent miners earnings (fees + issued coins) vs amount transacted in that interval
            cm_picked['earned_vs_transacted'] = (cm.isstotntv + cm.feetotntv) / cm.txtfrvaladjntv
        if 'isstotntv' in cm.columns:
            # isstotntv is total number of coins mined in the time interval
            # splycur is total number of coins mined (all time)
            total_mined = cm.isstotntv.rolling(365, min_periods=7).sum() # total mined in a year
            cm_picked['isstot365_isstot1_pct'] = (total_mined / cm.isstotntv).pct_change()
        if 'splycur' in cm.columns and 'isstotntv' in cm.columns:
            cm_picked['splycur_isstot1_pct'] = (cm.splycur / cm.isstotntv).pct_change()
        if 'hashrate' in cm.columns:
            #cm_picked['hashrate_mean3_pct'] = cm.hashrate.rolling(3).mean().pct_change()
            #cm_picked['hashrate_mean7_pct'] = cm.hashrate.rolling(7).mean().pct_change()
            cm_picked['hashrate_pct'] = cm.hashrate.pct_change()
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
        # REMA / RSMA are already used and well-estabilished in ATSA,
        # I'm taking the pct change since i want to encode the relative movement of the ema's not their positions
        # ta_picked['rema_5_20_pct'] = ta.rema_5_20.pct_change()
        ta_picked['rema_8_15_pct'] = ta.rema_8_15.pct_change()
        # ta_picked['rema_20_50_pct'] = ta.rema_20_50.pct_change()
        # ta_picked['rsma_5_20_pct'] = ta.rema_5_20.pct_change()
        ta_picked['rsma_8_15_pct'] = ta.rema_8_15.pct_change()
        # ta_picked['rsma_20_50_pct'] = ta.rema_20_50.pct_change()

        # Stoch is a momentum indicator comparing a particular closing price of a security to a range of its prices
        # over a certain period of time.
        # The sensitivity of the oscillator to market movements is reducible by adjusting that time period or
        # by taking a moving average of the result.
        # It is used to generate overbought and oversold trading signals, utilizing a 0-100 bounded range of values.
        # IDEA => decrease sensitivity by 3-mean and divide by 100 to get fp values
        ta_picked['stoch_14_mean3_div100'] = ta.stoch_14.rolling(3).mean() / 100

        #Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows
        # the relationship between two moving averages of a security’s price.
        # The MACD is calculated by subtracting the 26-period Exponential Moving Average (EMA) from the 12-period EMA.
        #  A nine-day EMA of the MACD called the "signal line," is then plotted on top of the MACD line,
        #  which can function as a trigger for buy and sell signals.
        #  Traders may buy the security when the MACD crosses above its signal line and sell - or short - the security
        #  when the MACD crosses below the signal line.
        #  Moving Average Convergence Divergence (MACD) indicators can be interpreted in several ways,
        #  but the more common methods are crossovers, divergences, and rapid rises/falls.
        signal_line = builder.exponential_moving_average(ta.macd_12_26, 9)
        ta_picked['macd_12_26_signal'] = (ta.macd_12_26 - signal_line).pct_change() # Relationship with signal line
        ta_picked['macd_12_26_pct'] = ta.macd_12_26.pct_change() # Information about slope

        # PPO is identical to the moving average convergence divergence (MACD) indicator,
        # except the PPO measures percentage difference between two EMAs, while the MACD measures absolute (dollar) difference.
        signal_line = builder.exponential_moving_average(ta.ppo_12_26, 9)
        ta_picked['ppo_12_26_signal'] = (ta.ppo_12_26 - signal_line).pct_change()  # Relationship with signal line
        ta_picked['ppo_12_26_pct'] = ta.ppo_12_26.pct_change()  # Information about slope

        # ADI Accumulation/distribution is a cumulative indicator that uses volume and price to assess whether
        # a stock is being accumulated or distributed.
        # The accumulation/distribution measure seeks to identify divergences between the stock price and volume flow.
        # This provides insight into how strong a trend is. If the price is rising but the indicator is falling
        # this indicates that buying or accumulation volume may not be enough to support
        # the price rise and a price decline could be forthcoming.
        # ==> IDEA: if we can fit a line to the price y1 = m1X+q1 and a line to ADI y2=m2X+q2 then we can identify
        #           divergences by simply looking at the sign of M.
        #           Another insight would be given by the slope (ie pct_change)
        ta_picked['adi_pct'] = ta.adi.pct_change()
        ta_picked['adi_close_convergence'] = convergence_between_series(ta.adi, ohlcv.close, 3)

        # RSI goes from 0 to 100, values <= 20 mean BUY, while values >= 80 mean SELL.
        # Dividing it by 100 to get a floating point feature, makes no sense to pct_change it
        ta_picked['rsi_14_div100'] = ta.rsi_14 / 100

        # The Money Flow Index (MFI) is a technical indicator that generates overbought or oversold
        #   signals using both prices and volume data. The oscillator moves between 0 and 100.
        # An MFI reading above 80 is considered overbought and an MFI reading below 20 is considered oversold,
        #   although levels of 90 and 10 are also used as thresholds.
        # A divergence between the indicator and price is noteworthy. For example, if the indicator is rising while
        #   the price is falling or flat, the price could start rising.
        ta_picked['mfi_14_div100'] = ta.mfi_14 / 100

        # The Chande momentum oscillator is a technical momentum indicator similar to other momentum indicators
        #   such as Wilder’s Relative Strength Index (Wilder’s RSI) and the Stochastic Oscillator.
        #   It measures momentum on both up and down days and does not smooth results, triggering more frequent
        #   oversold and overbought penetrations. The indicator oscillates between +100 and -100.
        # Many technical traders add a 10-period moving average to this oscillator to act as a signal line.
        #   The oscillator generates a bullish signal when it crosses above the moving average and a
        #   bearish signal when it drops below the moving average.
        ta_picked['cmo_14_div100'] = ta.cmo_14 / 100
        signal_line = builder.simple_moving_average(ta.cmo_14, 10)
        ta_picked['cmo_14_signal'] = (ta.cmo_14 - signal_line) / 100

        # On-balance volume (OBV) is a technical trading momentum indicator that uses volume flow to predict changes in stock price.
        # Eventually, volume drives the price upward. At that point, larger investors begin to sell, and smaller investors begin buying.
        # Despite being plotted on a price chart and measured numerically,
        # the actual individual quantitative value of OBV is not relevant.
        # The indicator itself is cumulative, while the time interval remains fixed by a dedicated starting point,
        # meaning the real number value of OBV arbitrarily depends on the start date.
        # Instead, traders and analysts look to the nature of OBV movements over time;
        # the slope of the OBV line carries all of the weight of analysis. => We want percent change
        ta_picked['obv_pct'] = ta.obv.pct_change()
        ta_picked['obv_mean3_pct'] = ta.obv.rolling(3).mean().pct_change()

        # Strong rallies in price should see the force index rise.
        # During pullbacks and sideways movements, the force index will often fall because the volume
        # and/or the size of the price moves gets smaller.
        # => Encoding the percent variation could be a good idea
        ta_picked['fi_13_pct'] = ta.fi_13.pct_change()
        ta_picked['fi_50_pct'] = ta.fi_50.pct_change()

        # The Aroon Oscillator is a trend-following indicator that uses aspects of the
        # Aroon Indicator (Aroon Up and Aroon Down) to gauge the strength of a current trend
        # and the likelihood that it will continue.
        # It moves between -100 and 100. A high oscillator value is an indication of an uptrend
        # while a low oscillator value is an indication of a downtrend.
        ta_picked['ao_14'] = ta.ao_14 / 100

        # The average true range (ATR) is a technical analysis indicator that measures market volatility
        #   by decomposing the entire range of an asset price for that period.
        # ATRP is pct_change of volatility
        ta_picked['atrp_14'] = ta.atrp_14

        # Percentage Volume Oscillator (PVO) is momentum volume oscillator used in technical analysis
        #   to evaluate and measure volume surges and to compare trading volume to the average longer-term volume.
        # PVO does not analyze price and it is based solely on volume.
        #  It compares fast and slow volume moving averages by showing how short-term volume differs from
        #  the average volume over longer-term.
        #  Since it does not care a trend's factor in its calculation (only volume data are used)
        #  this technical indicator cannot be used alone to predict changes in a trend.
        ta_picked['pvo_12_26'] = ta.pvo_12_26

        # IGNORED: tsi, wd, adx,

        #lagged_stats = pd.concat([ohlcv_stats] + [builder.make_lagged(ohlcv_stats, i) for i in range(1,10+1)], axis='columns', verify_integrity=True, sort=True, join='inner')

        # Build the dataframe with base features
        # lagged_close = pd.concat([ohlcv.close.pct_change()] + [builder.make_lagged(ohlcv.close.pct_change(), i) for i in range(1,10+1)], axis='columns', verify_integrity=True, sort=True, join='inner')
        # lagged_close.columns = ['close_pct'] + ['close_pct_lag-{}'.format(i) for i in range(1, W +1)]

        ohlc = ohlcv[['close', 'volume']].pct_change()
        lagged_ohlc = pd.concat([ohlc] + [builder.make_lagged(ohlc, i) for i in range(1, W + 1)], axis='columns',
                                verify_integrity=True, sort=True, join='inner')

        # Add lagged features to the dataframe
        improved_df = pd.concat([ohlcv_stats, lagged_ohlc, cm_picked, ta_picked], axis='columns', verify_integrity=True, sort=True, join='inner')

        # Drop the first 30 rows
        improved_df = improved_df[30:]
        # Drop columns whose values are all nan or inf
        with pd.option_context('mode.use_inf_as_na', True): # Set option temporarily
            improved_df = improved_df.dropna(axis='columns', how='all')
        # Save the dataframe
        improved_df.to_csv('data/datasets/all_merged/csv/{}_improved.csv'.format(_sym.lower()), sep=',', encoding='utf-8', index=True,
                  index_label='Date')
        improved_df.to_excel('data/datasets/all_merged/excel/{}_improved.xlsx'.format(_sym.lower()), index=True, index_label='Date')
        unlagged_df = improved_df.loc[ :,[ c for c in improved_df.columns if not '_lag' in c]]
        unlagged_df['target_pct'] = _target.loc[improved_df.index]['pct']
        unlagged_df['target_binary_bin'] = _target.loc[improved_df.index]['binary_bin']
        plot_correlation_matrix(unlagged_df.corr(), unlagged_df.columns, title='{} Correlation matrix'.format(_sym), save_to='data/datasets/all_merged/{}_improved_corr.png'.format(_sym))
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

def build_faceted_dataset(source_index, W=10):
    _dataset = load_dataset(source_index, return_index=True)
    index = {}

    for _sym, entry in _dataset.items():
        _df = pd.read_csv(entry['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        _target = pd.read_csv(entry['target_csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)

        ta = _df[entry['features']['ta']]
        cm = _df[entry['features']['cm']]

        # Price history facet (Daily variation of ohlc in last W trading days)
        ohlc = _df[['open', 'high', 'low', 'close']].pct_change()
        ohlc.columns = ['open_pct', 'high_pct', 'low_pct', 'close_pct']
        history_facet = pd.concat([ohlc] + [builder.make_lagged(ohlc, i) for i in range(1, W + 1)], axis='columns',
                            verify_integrity=True, sort=True, join='inner')
        # Price trend facet (REMA/RSMA, MACD, AO, ADX, WD+ - WD-)
        trend_facet = ta[["rsma_5_20", "rsma_8_15", "rsma_20_50", "rema_5_20", "rema_8_15", "rema_20_50", "macd_12_26", "ao_14", "adx_14", "wd_14"]]
        # Volatility facet (CMO, ATRp)
        volatility_facet = ta[["cmo_14", "atrp_14"]]
        # Volume facet (Volume pct, PVO, ADI, OBV)
        volume_facet = pd.concat([_df.volume.pct_change().replace([np.inf, -np.inf], 0), ta[["pvo_12_26", "adi", "obv"]]], axis='columns', verify_integrity=True, sort=True, join='inner')
        # On-chain facet
        cm_1 = cm.reindex(columns=['adractcnt', 'txtfrvaladjntv', 'isstotntv', 'feetotntv', 'splycur', 'hashrate', 'txtfrcount']).pct_change()
        cm_2 = cm.reindex(columns=['isscontpctann'])
        chain_facet = pd.concat([cm_1, cm_2], axis='columns', verify_integrity=True, sort=True, join='inner')

        # Drop columns whose values are all nan or inf from each facet
        with pd.option_context('mode.use_inf_as_na', True): # Set option temporarily
            history_facet = history_facet.dropna(axis='columns', how='all')
            trend_facet = trend_facet.dropna(axis='columns', how='all')
            volatility_facet = volatility_facet.dropna(axis='columns', how='all')
            volume_facet = volume_facet.dropna(axis='columns', how='all')
            chain_facet = chain_facet.dropna(axis='columns', how='all')

        improved_df = pd.concat([history_facet, trend_facet, volatility_facet, volume_facet, chain_facet],
                                axis='columns', verify_integrity=True, sort=True, join='inner')
        # Drop the first 30 rows
        improved_df = improved_df[30:]
        # Save the dataframe
        improved_df.to_csv('data/datasets/all_merged/csv/{}_faceted.csv'.format(_sym.lower()), sep=',', encoding='utf-8', index=True,
                  index_label='Date')
        improved_df.to_excel('data/datasets/all_merged/excel/{}_faceted.xlsx'.format(_sym.lower()), index=True, index_label='Date')

        # Add symbol to index
        index[_sym] = {
            'csv': 'data/datasets/all_merged/csv/{}_faceted.csv'.format(_sym.lower()),
            'xls': 'data/datasets/all_merged/excel/{}_faceted.xlsx'.format(_sym.lower()),
            'target_csv': 'data/datasets/all_merged/csv/{}_target.csv'.format(_sym.lower()),
            'target_xls': 'data/datasets/all_merged/excel/{}_target.xlsx'.format(_sym.lower()),
            'features': {
                'price_history': [c for c in history_facet.columns],
                'trend': [c for c in trend_facet.columns],
                'volatility': [c for c in volatility_facet.columns],
                'volume': [c for c in volume_facet.columns],
                'chain': [c for c in chain_facet.columns],
            }
        }
        logger.info('Saved {} in data/datasets/all_merged/'.format(_sym))
    with open('data/datasets/all_merged/index_faceted.json', 'w') as f:
        json.dump(index, f, sort_keys=True, indent=4)

if __name__ == '__main__':
    logger.setup(
        filename='../build_dataset.log',
        filemode='w',
        root_level=logging.DEBUG,
        log_level=logging.DEBUG,
        logger='build_dataset'
    )
    #build_merged_dataset()
    #build_atsa_dataset('all_merged')
    #build_improved_dataset('all_merged')
    build_faceted_dataset('all_merged')