import numpy as np
from keras.utils import to_categorical
from collections import deque
import pandas as pd

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