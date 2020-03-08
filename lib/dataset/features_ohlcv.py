import pandas as pd
from lib.utils import to_discrete_single, to_discrete_double

DEFAULT_MAP = {
    'open' : 'open',
    'high' : 'high',
    'low' : 'low',
    'close' : 'close',
    'volume' : 'volume'
}

def features_ohlcv(df, **kwargs):
	_map = kwargs.get('column_map', DEFAULT_MAP)
	if kwargs.get('symbol'):
		_map = _map_symbol(kwargs.get('symbol'))
	ohlcv = df[list(_map.values())].copy()
	ohlcv.columns = list(_map.keys())
	# Slice dataframe so that only valid indexes are kept
	ohlcv = ohlcv.loc[ohlcv.first_valid_index():ohlcv.last_valid_index()]
	# Check for rows containing NAN values
	#null_data = ohlcv.isnull()  # Returns dataframe mask where true represents missing value
	# Drop nan values
	ohlcv.dropna(axis='index', how='any', inplace=True)
	# Determine target (Next day close so shift 1 backwards)
	#target = ohlcv.close.shift(-1)  # Index is already taken care of.
	return ohlcv

def _map_symbol(_sym):
	return {
		'open': _sym + '_Open',
		'high': _sym + '_High',
		'low': _sym + '_Low',
		'close': _sym,
		'volume': _sym + '_Volume'
	}