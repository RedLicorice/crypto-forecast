import pandas as pd
from .technical_indicators import *
from .utils import to_discrete_single, to_discrete_double, future, pct_variation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from enum import Enum
from datetime import datetime
import talib

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
        target = pd.Series(future(ohlcv['close'].values, 1), index=ohlcv.index)  # set to number of forecast periods
        self.datasets[DatasetType.OHLCV] = (ohlcv, target)
        return ohlcv

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
            variation[k] = pct_variation(ta[k], -1)
        continuous = self.scale(continuous)
        variation = self.scale(variation, scaler=MinMaxScaler())
        # continuous.dropna(axis='index', how='any', inplace=True)
        # discrete.dropna(axis='index', how='any', inplace=True)

        pct_var = pct_variation(ohlcv['close'].values, 1)  # set to number of forecast periods
        classes = to_discrete_double(pct_var, -0.01, 0.01)

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
        price_pct= ohlcv.copy()
        for c in price_pct.columns:
            price_pct[c] = pct_variation(price_pct[c])
        # price_pct['target'] = future(price_pct['close'].values, 1)
        # self.features[DatasetType.OHLCV_PCT] = price_pct
        target = pd.Series(future(price_pct['close'].values, 1), index=price_pct.index)
        self.datasets[DatasetType.OHLCV_PCT] = (price_pct, target)
        return price_pct

    def build_blockchain_dataset(self, df, **kwargs):
        """
        :param df:
        :param kwargs:
        :return:

        === Run information ===

        Evaluator:    weka.attributeSelection.CorrelationAttributeEval
        Search:       weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1
        Relation:     blockchain-dataset
        Instances:    3322
        Attributes:   42
                      Date
                      AdrActCnt
                      BlkCnt
                      BlkSizeByte
                      BlkSizeMeanByte
                      CapMVRVCur
                      CapMrktCurUSD
                      CapRealUSD
                      DiffMean
                      FeeMeanNtv
                      FeeMeanUSD
                      FeeMedNtv
                      FeeMedUSD
                      FeeTotNtv
                      FeeTotUSD
                      IssContNtv
                      IssContPctAnn
                      IssContUSD
                      IssTotNtv
                      IssTotUSD
                      NVTAdj
                      NVTAdj90
                      PriceBTC
                      PriceUSD
                      ROI1yr
                      ROI30d
                      SplyCur
                      TxCnt
                      TxTfrCnt
                      TxTfrValAdjNtv
                      TxTfrValAdjUSD
                      TxTfrValMeanNtv
                      TxTfrValMeanUSD
                      TxTfrValMedNtv
                      TxTfrValMedUSD
                      TxTfrValNtv
                      TxTfrValUSD
                      VtyDayRet180d
                      VtyDayRet30d
                      VtyDayRet60d
                      target_price
                      target_pct
        Evaluation mode:    evaluate on all training data



        === Attribute Selection on all input data ===

        Search Method:
            Attribute ranking.

        Attribute Evaluator (supervised, Class (numeric): 42 target_pct):
            Correlation Ranking Filter
        Ranked attributes:
         0.066311    6 CapMVRVCur
         0.047588   26 ROI30d
         0.039514    3 BlkCnt
         0.03442    17 IssContPctAnn
         0.033876   19 IssTotNtv
         0.033876   16 IssContNtv
         0.029516   22 NVTAdj90
         0.028813   32 TxTfrValMeanNtv
         0.020041   30 TxTfrValAdjNtv
         0.018183   10 FeeMeanNtv
         0.017334   39 VtyDayRet30d
         0.01313    38 VtyDayRet180d
         0.007607    1 Date
         0.007331   34 TxTfrValMedNtv
         0.004698   12 FeeMedNtv
         0.004569   36 TxTfrValNtv
         0.000392   40 VtyDayRet60d
         0          23 PriceBTC
        -0.003147   37 TxTfrValUSD
        -0.00421    14 FeeTotNtv
        -0.006254   33 TxTfrValMeanUSD
        -0.017056   31 TxTfrValAdjUSD
        -0.017192   13 FeeMedUSD
        -0.017937   15 FeeTotUSD
        -0.018518   25 ROI1yr
        -0.020396   11 FeeMeanUSD
        -0.023797    2 AdrActCnt
        -0.025366   29 TxTfrCnt
        -0.026377   18 IssContUSD
        -0.026377   20 IssTotUSD
        -0.027841   28 TxCnt
        -0.028096   41 target_price
        -0.028458   21 NVTAdj
        -0.029616   24 PriceUSD
        -0.030107    7 CapMrktCurUSD
        -0.031475    4 BlkSizeByte
        -0.033649   35 TxTfrValMedUSD
        -0.033897    5 BlkSizeMeanByte
        -0.040972   27 SplyCur
        -0.046898    8 CapRealUSD
        -0.047285    9 DiffMean

        Selected attributes: 6,26,3,17,19,16,22,32,30,10,39,38,1,34,12,36,40,23,37,14,33,31,13,15,25,11,2,29,18,20,28,41,21,24,7,4,35,5,27,8,9 : 41

        === CFS Subset Evaluation w/ PCA ===

        Search Method:
            Best first.
            Start set: no attributes
            Search direction: forward
            Stale search after 5 node expansions
            Total number of subsets evaluated: 369
            Merit of best subset found:    0.072

        Attribute Subset Evaluator (supervised, Class (numeric): 42 target_pct):
            CFS Subset Evaluator
            Including locally predictive attributes

        Selected attributes: 1,6,9,17,32 : 5
                             Date
                             CapMVRVCur
                             DiffMean
                             IssContPctAnn
                             TxTfrValMeanNtv
        """
        ohlcv, ohlcv_target = self.datasets[DatasetType.OHLCV]
        pct, pct_target =  self.datasets[DatasetType.OHLCV_PCT]
        ta, ta_target =  self.datasets[DatasetType.DISCRETE_TA]

        if ohlcv is None or pct is None:
            raise RuntimeError('No ohlcv or ta.py loaded!')

        #reduced = df.dropna(axis='index')
        df = df.merge(ohlcv, how='left', left_index=True, right_index=True)
        scaled = self.scale(df, exclude=kwargs.get('exclude'))


        targets = pd.DataFrame(index=ohlcv.index)
        targets['target_price'] = self.scale(ohlcv_target)
        targets['target_pct'] = pct_target
        targets['target_class'] = ta_target

        result = pd.merge(scaled, targets, how='inner', left_index=True, right_index=True)

        self.datasets[DatasetType.BLOCKCHAIN] = (result, ta_target)
        return scaled

    ## Operands for classifiers
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

    def scale(self, df, **kwargs):
        scaler = kwargs.get('scaler', StandardScaler())
        # scaler selection by name
        if isinstance(scaler, str):
            if scaler == 'standard':
                scaler = StandardScaler()
            elif scaler == 'minmax':
                scaler = MinMaxScaler()
        # Dataframe transparent scaling
        if isinstance(df, pd.DataFrame):
            scaled = pd.DataFrame(index=df.index)
            columns = kwargs.get('columns', df.columns)
            exclude = kwargs.get('exclude', [])
            for c in columns:
                if exclude is not None and c in exclude:
                    scaled[c] = df[c].values
                    continue
                if str(df[c].dtype) == 'int64':
                    df[c] = df[c].astype(float) # Suppress int-to-float conversion warnings
                scaled[c] = scaler.fit_transform(np.reshape(df[c].values, (-1, 1)))
            return scaled
        elif isinstance(df, list) or isinstance(df, np.ndarray) or isinstance(df, pd.Series):
            if isinstance(df, pd.Series):
                df = df.values
            scaled = np.reshape(df, (-1,1))
            return scaler.fit_transform(scaled)

    ## Filters for data
    def time_slice(self, begin, end, **kwargs):
        format = kwargs.get('format', '%Y-%m-%d %H:%M:%S')
        begin = datetime.strptime(begin, format)
        end = datetime.strptime(end, format)

        result = Symbol(self.name)
        for type, (df, tgt) in self.datasets.items():
            result.datasets[type] = (df.loc[begin:end].copy(), tgt.loc[begin:end].copy())
        return result