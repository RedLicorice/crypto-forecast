import pandas as pd
from .technical_indicators import *
from .utils import to_discrete_single, to_discrete_double
from sklearn.preprocessing import StandardScaler
from enum import Enum
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

class Symbol:
    """
    Encapsulate relevant data for a given symbol
    ---
    - Build a dataset with technical features by using OHLCV data
    """
    datasets = None
    index = None
    name = None

    def __init__(self, name, **kwargs):
        self.name = name
        self.datasets = {}
        if kwargs.get('ohlcv') is not None:
            self.build_datasets(ohlcv=kwargs.get('ohlcv'), column_map=kwargs.get('column_map', DEFAULT_MAP))

    def __repr__(self):
        return 'symbol_{}'.format(self.name)

    ## Load DatasetType.OHLCV dataset
    def build_ohlcv_dataset(self, df, **kwargs):
        _map = kwargs.get('column_map', DEFAULT_MAP)
        # Drop all columns but OHLCV we're interested in (given in column_map),
        # then rename dataframe's columns to what we expect
        # self.ohlcv = df[list(_map.values())]
        # self.ohlcv.columns = list(_map.keys())
        # self.cont_ds =  pd.DataFrame(index=self.ohlcv.index) # Save the time series index
        # self.discr_ds =  pd.DataFrame(index=self.ohlcv.index)  # Save the time series index
        # self.price_pct_ds =  pd.DataFrame(index=self.ohlcv.index)  # Save the time series index
        ohlcv = df[list(_map.values())].copy()
        ohlcv.columns = list(_map.keys())
        ohlcv['target'] = self.future(ohlcv['close'].values, 1)  # set to number of forecast periods
        self.datasets[DatasetType.OHLCV] = ohlcv
        return ohlcv

    ## Build TA dataset
    def build_ta_dataset(self):
        ohlcv = self.datasets[DatasetType.OHLCV]
        if ohlcv is None:
            raise RuntimeError('No ohlcv loaded!')

        ta = self.get_ta_features(ohlcv['high'].values,
                                  ohlcv['low'].values,
                                  ohlcv['close'].values,
                                  ohlcv['volume'].values)
        dta = self.discretize_ta_features(ta)

        continuous = pd.DataFrame(index=ohlcv.index)
        discrete = pd.DataFrame(index=ohlcv.index)
        for k, dk in zip(ta.keys(), dta.keys()):  # Keys are the same both for 'ta' and 'dta'
            continuous[k] = ta[k]
            discrete[dk] = dta[dk]
        # continuous.dropna(axis='index', how='any', inplace=True)
        # discrete.dropna(axis='index', how='any', inplace=True)

        pct_var = self.pct_variation(ohlcv['close'].values, 1)  # set to number of forecast periods
        classes = to_discrete_double(pct_var, -0.01, 0.01)
        continuous['target'] = pct_var
        discrete['target'] = classes

        self.datasets[DatasetType.CONTINUOUS_TA] = continuous
        self.datasets[DatasetType.DISCRETE_TA] = discrete

    def build_ohlcv_pct_dataset(self):
        ohlcv = self.datasets[DatasetType.OHLCV]
        if ohlcv is None:
            raise RuntimeError('No ohlcv loaded!')
        # Price pct dataset is well, price from ohlcv but in percent variations
        price_pct= ohlcv.copy()
        for c in price_pct.columns:
            price_pct[c] = self.pct_variation(price_pct[c])
        price_pct['target'] = self.future(price_pct['close'].values, 1)

        self.datasets[DatasetType.OHLCV_PCT] = price_pct

    def build_datasets(self, ohlcv, **kwargs):
        """
        Build discrete dataset (for classification tasks), continuous
        dataset (for regression tasks) and price dataset
        """
        self.build_ohlcv_dataset(ohlcv, column_map=kwargs.get('column_map', DEFAULT_MAP))
        self.build_ta_dataset() # Requires ohlcv dataset
        self.build_ohlcv_pct_dataset() # Requires ohlcv dataset

        # if scaler
        scale_types = kwargs.get('scale')
        scaler = StandardScaler()
        if scale_types:
            for type in scale_types:
                if type in self.datasets:
                    ds = self.datasets[type]
                    for col in ds.columns.difference(kwargs.get('scaler-exclude',[])):
                        reshaped = np.reshape(ds[col].values, (-1,1))
                        self.datasets[type][col] = scaler.fit_transform(reshaped)

        # Add input lag if needed
        # if kwargs.get('lag'):
        #     self.discr_ds = add_lag(self.discr_ds, kwargs.get('lag'))
        #     self.cont_ds = add_lag(self.cont_ds, kwargs.get('lag'))


    ## Operands for classifiers
    def get_xy(self, type):
        if not type in self.datasets:
            raise RuntimeError('Dataset type {} not loaded!'.format(type))
        ds = self.datasets[type]
        return ds[ds.columns.difference(['target'])], ds[['target']]

    def get_dataset(self, type):
        if not type in self.datasets:
            raise RuntimeError('Dataset type {} not loaded!'.format(type))
        return self.datasets[type]

    ## Synthetic features
    def get_ta_features(self, high, low, close, volume):
        """
        Returns a dict containing the technical analysis indicators calculated on the given
        high, low, close and volumes.
        """
        ta = {}
        
        # Set numpy to ignore division error and invalid values (since not all datasets are complete)
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
        # ta['stoch'] = percent_k(high, low, close, 14)

        # Chande Momentum Oscillator
        ## Not available in ta
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

    ## Target
    def pct_variation(self, close, periods = 1):
        """
        Calculate percent change with next N-th period using formula:
            next_close - close / close
        """
        next_close = self.future(close, periods)
        return np.divide(next_close - close, close)

    def future(self, x, periods):
        next_close = np.roll(x, -periods)
        for i in range(periods):
            next_close[-i] = 0.0
        return next_close
