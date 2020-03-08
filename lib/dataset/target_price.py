from lib.dataset import DatasetFactory


def build(ohlcv, **kwargs):
    if not 'close' in ohlcv.columns:
        raise ValueError("Input is not valid OHLCV data!")
    return ohlcv.close.shift(-kwargs.get('periods', 1))

DatasetFactory.register_target('price', build)