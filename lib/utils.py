import numpy as np
#from keras.utils import to_categorical
from collections import deque
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def check_duplicates(df, **kwargs):
    present = []
    duplicates = []
    for i in df.index:
        if not i in present:
            present.append(i)
        else:
            if kwargs.get('print'):
                print('Duplicate index: ' + str(i))
            duplicates.append(i)
    return duplicates

def is_constant(iterable):
    # Idea is to get the unique values in the array, and check how many of them there are.
    # One unique value => array is constant
    if isinstance(iterable, pd.DataFrame) or isinstance(iterable, pd.Series):
        iterable = iterable.values
    _clss = np.unique(iterable)
    return _clss.shape[0] <= 1

def has_negative(iterable):
    if isinstance(iterable, pd.DataFrame) or isinstance(iterable, pd.Series):
        iterable = iterable.values
    iterable = np.reshape(iterable, (-1,1))
    return len([i for i in iterable if i < 0]) > 0

def scale(df, **kwargs):
    scaler = kwargs.get('scaler', StandardScaler())
    # scaler selection by name
    if isinstance(scaler, str):
        if scaler == 'minmax':
            scaler = MinMaxScaler(feature_range=kwargs.get('feature_range', (0,1)))
        else:  # Default scaler
            scaler = StandardScaler()
    # Dataframe transparent scaling
    if isinstance(df, pd.DataFrame):
        scaled = pd.DataFrame(index=df.index)
        columns = kwargs.get('columns', df.columns)
        exclude = kwargs.get('exclude', []) or []
        for c in columns:
            if c in exclude:
                scaled[c] = df[c].values
                continue
            if str(df[c].dtype) == 'int64':
                df[c] = df[c].astype(float)  # Suppress int-to-float conversion warnings
            scaled[c] = scaler.fit_transform(np.reshape(df[c].values, (-1, 1)))
        return scaled
    elif isinstance(df, list):
        return scaler.fit_transform(df)
    elif isinstance(df, pd.Series):
        values = np.reshape(df.values, (-1,1))
        scaled = scaler.fit_transform(values)
        scaled = np.reshape(scaled, (-1,))
        return pd.Series(scaled, index=df.index)
    elif isinstance(df, np.ndarray):
        old_shape = df.shape
        values = np.reshape(df, (-1,1))
        scaled = scaler.fit_transform(values)
        return np.reshape(scaled, old_shape)


def oversample(X, y):
    sm = SMOTE(random_state=12)
    rX, rY =  sm.fit_sample(X, y)
    if isinstance(X, pd.DataFrame):
        rX = pd.DataFrame(data=rX, columns=X.columns)
    elif isinstance(X, pd.Series):
        rX = pd.Series(data=rX)
    if isinstance(y, pd.Series):
        rY = pd.Series(data=rY)
    return rX, rY


def undersample(X, y):
    cc = ClusterCentroids(random_state=12)
    rX, rY = cc.fit_resample(X, y)
    if isinstance(X, pd.DataFrame):
        rX = pd.DataFrame(data=rX, columns=X.columns)
    elif isinstance(X, pd.Series):
        rX = pd.Series(data=rX)
    if isinstance(y, pd.Series):
        rY = pd.Series(data=rY)
    return rX, rY

def add_lag(df, lag, exclude = None):
    if exclude is None:
        exclude = []
    lags = range(1,lag)
    return df.assign(**{
        '{}_-{}'.format(col, t): df[col].shift(t)
        for t in lags
        for col in df
        if col not in exclude
    })

def future(x, periods):
    next_close = np.roll(x, -periods)
    for i in range(periods):
        next_close[-i] = 0.0
    return next_close

def pct_variation(x, periods = 1):
    """
    Calculate percent change with next N-th period using formula:
        next_close - close / close
    """
    next = future(x, periods)
    diff = next - x
    return np.divide(diff, x, where=x!=0) # , out=np.zeros_like(close))

def to_discrete_single(values, threshold):
    def _to_discrete(x, threshold):
        if np.isnan(x):
            return np.nan
        if x < threshold:
            return 1
        return 2

    fun = np.vectorize(_to_discrete)
    return fun(values, threshold)


def to_discrete_double(values, threshold_lo=0.01, threshold_hi=0.01):
    def _to_discrete(x, threshold_lo, threshold_hi):
        if np.isnan(x):
            return np.nan
        if x <= threshold_lo:
            return 1
        elif threshold_lo < x < threshold_hi:
            return 2
        else:
            return 3

    fun = np.vectorize(_to_discrete)
    return fun(values, threshold_lo, threshold_hi)


def discrete_label(values):
    def _to_label(cls):
        if np.isnan(cls):
            return np.nan
        if cls == 1:
            return 'Decrease'
        elif cls == 2:
            return 'Stationary'
        elif cls == 3:
            return 'Increase'

    return np.vectorize(_to_label)(values)

def from_categorical(encoded):
    res = []
    for i in range(encoded.shape[0]):
        datum = encoded[i]
        decoded = np.argmax(encoded[i])
        res.append(decoded)
    return np.array(res) + 1 # +1 because keras' from_categoric encodes from 0 while our classes start from 1

def to_categorical(classes):
    res = []
    _clss = np.unique(classes)
    for i in range(classes.shape[0]):
        datum = [0 for i in _clss]
        for j in range(len(_clss)):
            if classes[i] == _clss[j]:
                datum[j] = 1
        res.append(datum)
    return np.array(res) # +1 because keras' to_categoric encodes from 0 while our classes start from 1

def get_unique_ratio(arr):
    total = max(1,len(arr))
    unique, counts = np.unique(arr, return_counts=True)
    return {cls:(cnt, cnt*100/total) for cls,cnt in zip(unique,counts)}

def to_timesteps(X: pd.DataFrame, y: pd.DataFrame, ts: int, **kwargs):
    """
    Returns an X vector in shape (samples, timesteps, features)
        for use as input for a LSTM network.
    Y vector's index is the original index minus the first ts-1 steps,
        so original_index[ts-1,:]
    """
    tX, ty = [], []
    _stack = deque([], ts) # samples for features in current timestep
    for i in range(X.shape[0]):
        _stack.append(X[i])
        if len(_stack) < ts:
            continue
        tX.append(list(_stack.copy()))
        ty.append(y[i])

    return np.array(tX), np.array(ty), np.array(X.index[ts-1:])