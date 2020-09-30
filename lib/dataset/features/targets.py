import pandas as pd
import numpy as np
from lib.dataset.utils import to_discrete_double, to_discrete_single
from sklearn.preprocessing import KBinsDiscretizer

def target_price(close : pd.Series, **kwargs):
    return close.shift(-kwargs.get('periods', 1))

def target_price_variation(close : pd.Series, **kwargs):
    pct_var = pd.Series(np.roll(close.pct_change(periods=kwargs.get('periods', 1)), -kwargs.get('periods', 1)), index=close.index)
    return pct_var

def target_discrete_price_variation(pct_var : pd.Series, **kwargs):
    classes = to_discrete_double(pct_var.fillna(method='ffill'), -0.01, 0.01)
    return pd.Series(classes, index=pct_var.index)

def target_binary_price_variation(pct_var : pd.Series, **kwargs):
    classes = to_discrete_single(pct_var.fillna(method='ffill'), 0.00)
    return pd.Series(classes, index=pct_var.index)

def target_label(classes, **kwargs):
    _labels = ['SELL', 'HOLD', 'BUY']
    if 'labels' in kwargs:
        _labels = kwargs.get('labels')
    return pd.Series([_labels[int(c)] for c in classes], index=classes.index)

def target_binned_price_variation(pct_var : pd.Series, **kwargs):
    values = pct_var.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').values
    values = np.reshape(values, (-1, 1))
    discretizer = KBinsDiscretizer(n_bins=kwargs.get('n_bins',3), strategy='quantile', encode='ordinal')
    discrete = discretizer.fit_transform(values)
    return pd.Series(np.reshape(discrete, (-1,)), index=pct_var.index)

def target_binned_price_variation_kmeans(pct_var : pd.Series, **kwargs):
    values = pct_var.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').values
    values = np.reshape(values, (-1, 1))
    discretizer = KBinsDiscretizer(n_bins=kwargs.get('n_bins',3), strategy='kmeans', encode='ordinal')
    discrete = discretizer.fit_transform(values)
    return pd.Series(np.reshape(discrete, (-1,)), index=pct_var.index)