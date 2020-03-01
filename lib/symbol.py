import pandas as pd
from lib.technical_indicators import *
from lib.utils import to_discrete_single, to_discrete_double, scale
from enum import Enum
from datetime import datetime
import talib
from functools import reduce
#from collections import deque
# ToDO: try aggregating candles at 1, 7 and 30 days

# Default mapping used for extracting OHLCV data from dataframe
DEFAULT_MAP = {
    'open' : 'open',
    'high' : 'high',
    'low' : 'low',
    'close' : 'close',
    'volume' : 'volume'
}

class DatasetType(Enum): # Dataset Type
    OHLCV = 1
    CONTINUOUS_TA = 2
    DISCRETE_TA = 3
    OHLCV_PCT = 4
    BLOCKCHAIN = 5
    OHLCV_PATTERN = 6
    VARIATION_TA = 7

class Symbol:
    """
    Encapsulate relevant data for a given symbol
    """
    datasets = None
    index = None
    name = None

    def __init__(self, name, **kwargs):
        self.name = name
        self.datasets = {}
        if kwargs.get('ohlcv') is not None:
            self.build_datasets(ohlcv=kwargs.get('ohlcv'), blockchain=kwargs.get('blockchain'), column_map=kwargs.get('column_map', DEFAULT_MAP))

    def __repr__(self):
        return '{}'.format(self.name)

    ## Builder all dataset types
    def build_datasets(self, ohlcv, **kwargs):
        """
        Build discrete dataset (for classification tasks), continuous
        dataset (for regression tasks) and price dataset
        """
        self.build_ohlcv_dataset(ohlcv, column_map=kwargs.get('column_map', DEFAULT_MAP))
        self.build_ta_dataset() # Requires ohlcv dataset
        self.build_ohlcv_pct_dataset() # Requires ohlcv dataset
        self.build_pattern_dataset()
        bdf = kwargs.get('blockchain', None)
        if bdf is not None:
            #print('Build blockchain ds!')
            self.build_blockchain_dataset(bdf)

    # Builder for each dataset
    def build_ohlcv_dataset(self, df, **kwargs):
        _map = kwargs.get('column_map', DEFAULT_MAP)
        ohlcv = df[list(_map.values())].copy()
        ohlcv.columns = list(_map.keys())
        #Slice dataframe so that only valid indexes are kept
        ohlcv = ohlcv.loc[ohlcv.first_valid_index():ohlcv.last_valid_index()]
        # Check for rows containing NAN values
        null_data = ohlcv.isnull() # Returns dataframe mask where true represents missing value
        # Drop nan values
        ohlcv.dropna(axis='index', how='any', inplace=True)
        #Determine target (Next day close so shift 1 backwards)
        target = ohlcv.close.shift(-1) # Index is already taken care of.
        self.datasets[DatasetType.OHLCV] = (ohlcv, target)
        self.build_higher_tfs()
        return ohlcv

    def build_higher_tfs(self):
        ohlcv, ohlcv_target = self.datasets[DatasetType.OHLCV]
        if ohlcv is None:
            raise RuntimeError('No ohlcv loaded!')
        #periods = [7,14,30]
        #queues = {n:deque([], n) for n in periods}
        #for i in ohlcv.index:
        #    for q in queues.values():
        #        q.append()
        weekly = ohlcv.resample('W')\
            .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        monthly = ohlcv.resample('M')\
            .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        return


    def build_pattern_dataset(self):
        ohlcv, ohlcv_target = self.datasets[DatasetType.OHLCV]
        if ohlcv is None:
            raise RuntimeError('No ohlcv loaded!')
        functions = talib.get_function_groups()['Pattern Recognition']

        patterns = pd.DataFrame(index=ohlcv.index)

        for fname in functions:
            f = getattr(talib, fname)
            # findings is a list whose length corresponds to passed values,
            # The i-th element of said list has value:
            # +100: bullish pattern detected in i-th position
            # 0: pattern not detected in i-th position
            # -100: bearish pattern detected in i-th position
            findings = f(ohlcv['open'].values, ohlcv['high'].values, ohlcv['low'].values, ohlcv['close'].values)
            # Discretize values according to label in utils.to_discrete_double
            patterns[fname] = [3 if v > 0 else 2 if v == 0 else 1 for v in findings]

        # patterns['target'] = self.features[DatasetType.DISCRETE_TA]['target']
        # self.features[DatasetType.OHLCV_PATTERN] = patterns
        target = self.datasets[DatasetType.DISCRETE_TA][1]
        self.datasets[DatasetType.OHLCV_PATTERN] = (patterns, target)

    def build_ta_dataset(self):
        ohlcv, ohlcv_target = self.datasets[DatasetType.OHLCV]
        if ohlcv is None:
            raise RuntimeError('No ohlcv loaded!')

        ta = self.get_ta_features(ohlcv['high'].values,
                                  ohlcv['low'].values,
                                  ohlcv['close'].values,
                                  ohlcv['volume'].values)
        dta = self.discretize_ta_features(ta)

        continuous = pd.DataFrame(index=ohlcv.index)
        discrete = pd.DataFrame(index=ohlcv.index)
        variation = pd.DataFrame(index=ohlcv.index)
        for k, dk in zip(ta.keys(), dta.keys()):  # Keys are the same both for 'ta' and 'dta'
            continuous[k] = ta[k]
            discrete[dk] = dta[dk]
            #variation[k] = pct_variation(ta[k], -1)
            variation[k] = continuous[k].pct_change(periods=1)

        #pct_var = pct_variation(ohlcv['close'].values, 1)  # set to number of forecast periods
        pct_var = pd.Series(np.roll(ohlcv.close.pct_change(periods=1), -1), index=ohlcv.index)
        classes = to_discrete_double(pct_var.fillna(method='ffill'), -0.01, 0.01)

        ## For debugging ds
        # continuous['close'] = ohlcv['close'].values
        # continuous['next_close'] = future(ohlcv['close'].values, 1)
        # continuous['discrete'] = classes
        # continuous['target'] = pct_var
        # discrete['target'] = classes
        # self.features[DatasetType.CONTINUOUS_TA] = continuous
        # self.features[DatasetType.DISCRETE_TA] = discrete

        self.datasets[DatasetType.CONTINUOUS_TA] = (continuous, pd.Series(pct_var, index=continuous.index))
        self.datasets[DatasetType.VARIATION_TA] = (variation, pd.Series(pct_var, index=continuous.index))
        self.datasets[DatasetType.DISCRETE_TA] = (discrete, pd.Series(classes, index=discrete.index))

        return continuous, discrete

    def build_ohlcv_pct_dataset(self):
        ohlcv, ohlcv_target = self.datasets[DatasetType.OHLCV]
        if ohlcv is None:
            raise RuntimeError('No ohlcv loaded!')
        # Price pct dataset is well, price from ohlcv but in percent variations
        price_pct= ohlcv.pct_change(1)
        #for c in price_pct.columns:
            #price_pct[c] = pct_variation(price_pct[c])
        # price_pct['target'] = future(price_pct['close'].values, 1)
        # self.features[DatasetType.OHLCV_PCT] = price_pct
        target = pd.Series(np.roll(ohlcv.close.pct_change(periods=1), -1), index=ohlcv.index)
        self.datasets[DatasetType.OHLCV_PCT] = (price_pct, target)
        return price_pct

    def build_blockchain_dataset(self, df, **kwargs):
        ohlcv, ohlcv_target = self.datasets[DatasetType.OHLCV]
        pct, pct_target =  self.datasets[DatasetType.OHLCV_PCT]
        ta, ta_target =  self.datasets[DatasetType.DISCRETE_TA]

        if ohlcv is None or pct is None:
            raise RuntimeError('No ohlcv or ta.py loaded!')

        #reduced = df.dropna(axis='index')
        df = df.merge(ohlcv, how='left', left_index=True, right_index=True)
        #df = df.pct_change(periods=1)
        df = df.diff(axis=0, periods=1)
        df = scale(df)
        #df = df.dropna(axis='index', how='any')
        df = df.loc[df.first_valid_index():df.last_valid_index()]
        df = df.interpolate(axis=1, method='linear')

        targets = pd.DataFrame(index=ohlcv.index)
        targets['target_price'] = scale(ohlcv_target)
        targets['target_pct'] = pct_target
        targets['target_class'] = ta_target

        result = pd.merge(df, targets, how='inner', left_index=True, right_index=True)
        # Drop values for which target is nan
        result = result.dropna(axis='index', subset=['target_price'])

        self.datasets[DatasetType.BLOCKCHAIN] = (result, ta_target)
        return result

    ## Operands for classifiers
    def get_x(self, type):
        x, y = self.get_xy(type)
        return x

    def get_y(self, type):
        x, y = self.get_xy(type)
        return y

    def get_xy(self, type):
        if not type in self.datasets:
            raise RuntimeError('Dataset type {} not loaded!'.format(type))
        ds, target = self.datasets[type]
        return ds, target

    def get_dataset(self, type):
        if not type in self.datasets:
            raise RuntimeError('Dataset type {} not loaded!'.format(type))
        ds, target = self.datasets[type]
        res = ds.copy()
        res['target'] = target
        return res

    ## Synthetic features
    def get_ta_features(self, high, low, close, volume):
        """
        Returns a dict containing the technical analysis indicators calculated on the given
        high, low, close and volumes.
        """
        ta = {}

        # Set numpy to ignore division error and invalid values (since not all features are complete)
        old_settings = np.seterr(divide='ignore', invalid='ignore')

        # Determine relative moving averages
        ta['rsma5_20'] = relative_sma(close, 5, 20)
        ta['rsma8_15'] = relative_sma(close, 8, 15)
        ta['rsma20_50'] = relative_sma(close, 20, 50)
        ta['rema5_20'] = relative_ema(close, 5, 20)
        ta['rema8_15'] = relative_ema(close, 8, 15)
        ta['rema20_50'] = relative_ema(close, 20, 50)

        # MACD Indicator
        ta['macd_12_26'] = moving_average_convergence_divergence(close, 12, 26)

        # Aroon Indicator
        ta['ao'] = aroon_oscillator(close, 14)

        # Average Directional Movement Index (ADX)
        ta['adx'] = average_directional_index(close, high, low,
                                                        14)

        # Difference between Positive Directional Index(DI+) and Negative Directional Index(DI-)
        ta['wd'] = \
            positive_directional_index(close, high, low, 14) \
            - negative_directional_index(close, high, low, 14)

        # Percentage Price Oscillator
        ta['ppo'] = price_oscillator(close, 12, 26)

        # Relative Strength Index
        ta['rsi'] = relative_strength_index(close, 14)

        # Money Flow Index
        ta['mfi'] = money_flow_index(close, high, low, volume, 14)

        # True Strength Index
        ta['tsi'] = true_strength_index(close)

        # Stochastic Oscillator
        ta['stoch'] = percent_k(close, 14)
        # ta.py['stoch'] = percent_k(high, low, close, 14)

        # Chande Momentum Oscillator
        ## Not available in ta.py
        ta['cmo'] = chande_momentum_oscillator(close, 14)

        # Average True Range Percentage
        ta['atrp'] = average_true_range_percent(close, 14)

        # Percentage Volume Oscillator
        ta['pvo'] = volume_oscillator(volume, 12, 26)

        # Force Index
        fi = force_index(close, volume)
        ta['fi13'] = exponential_moving_average(fi, 13)
        ta['fi50'] = exponential_moving_average(fi, 50)

        # Accumulation Distribution Line
        ta['adi'] = accumulation_distribution(close, high, low,
                                                        volume)

        # On Balance Volume
        ta['obv'] = on_balance_volume(close, volume)

        # Restore numpy error settings
        np.seterr(**old_settings)

        return ta

    def discretize_ta_features(self, ta):
        """
        Returns a dict containing the discretized version of the input
        dict of technical indicators
        """
        dta = {}
        dta['rsma5_20'] = to_discrete_single(ta['rsma5_20'], 0)
        dta['rsma8_15'] = to_discrete_single(ta['rsma8_15'], 0)
        dta['rsma20_50'] = to_discrete_single(ta['rsma20_50'], 0)
        dta['rema5_20'] = to_discrete_single(ta['rema5_20'], 0)
        dta['rema8_15'] = to_discrete_single(ta['rema8_15'], 0)
        dta['rema20_50'] = to_discrete_single(ta['rema20_50'], 0)
        dta['macd_12_26'] = to_discrete_single(ta['macd_12_26'], 0)
        dta['ao'] = to_discrete_single(ta['ao'], 0)
        dta['adx'] = to_discrete_single(ta['adx'], 20)
        dta['wd'] = to_discrete_single(ta['wd'], 0)
        dta['ppo'] = to_discrete_single(ta['ppo'], 0)
        dta['rsi'] = to_discrete_double(ta['rsi'], 30, 70)
        dta['mfi'] = to_discrete_double(ta['mfi'], 30, 70)
        dta['tsi'] = to_discrete_double(ta['tsi'], -25, 25)
        dta['stoch'] = to_discrete_double(ta['stoch'], 20, 80)
        dta['cmo'] = to_discrete_double(ta['cmo'], -50, 50)
        dta['atrp'] = to_discrete_single(ta['atrp'], 30)
        dta['pvo'] = to_discrete_single(ta['pvo'], 0)
        dta['fi13'] = to_discrete_single(ta['fi13'], 0)
        dta['fi50'] = to_discrete_single(ta['fi50'], 0)
        dta['adi'] = to_discrete_single(ta['adi'], 0)
        dta['obv'] = to_discrete_single(ta['obv'], 0)
        for k in dta.keys():
            dta[k] = [np.nan if np.isnan(x) else np.asscalar(x) for x in dta[k]]
        return dta

    ## Filters for data
    def time_slice(self, begin, end, **kwargs):
        format = kwargs.get('format', '%Y-%m-%d %H:%M:%S')
        begin = datetime.strptime(begin, format)
        end = datetime.strptime(end, format)

        result = Symbol(self.name)
        for type, (df, tgt) in self.datasets.items():
            result.datasets[type] = (df.loc[begin:end].copy(), tgt.loc[begin:end].copy())
        return result

    def add_lag(self, periods):
        result = Symbol(self.name)
        for type, (df, tgt) in self.datasets.items():
            lags = [df] + [df.shift(i).add_suffix('-{}'.format(i)) for i in range(1, periods+1)]
            lagged_df = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), lags)
            result.datasets[type] = (lagged_df, tgt)
        return result