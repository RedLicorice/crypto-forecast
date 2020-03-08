from lib.dataset import DatasetFactory
import pandas as pd
import numpy as np

def build(ohlcv, **kwargs):
    if not 'close' in ohlcv.columns:
        raise ValueError("Input is not valid OHLCV data!")
    pct_var = pd.Series(np.roll(ohlcv.close.pct_change(periods=kwargs.get('periods', 1)), -kwargs.get('periods', 1)), index=ohlcv.index)
    return pct_var

DatasetFactory.register_target('price_variation', build)