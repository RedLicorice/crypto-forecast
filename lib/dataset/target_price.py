

def target_price(ohlcv, **kwargs):
    if not 'close' in ohlcv.columns:
        raise ValueError("Input is not valid OHLCV data!")
    return ohlcv.close.shift(-kwargs.get('periods', 1))