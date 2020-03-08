from lib.dataset import DatasetFactory
import pandas as pd
import numpy as np
from lib.utils import to_discrete_double

def build(ohlcv, **kwargs):
    if not 'close' in ohlcv.columns:
        raise ValueError("Input is not valid OHLCV data!")
    pct_var = pd.Series(np.roll(ohlcv.close.pct_change(periods=kwargs.get('periods', 1)), -kwargs.get('periods', 1)), index=ohlcv.index)
    classes = to_discrete_double(pct_var.fillna(method='ffill'), -0.01, 0.01)
    return classes

DatasetFactory.register_target('discrete_price_variation', build)