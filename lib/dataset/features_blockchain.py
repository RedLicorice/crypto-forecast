import pandas as pd
from lib.utils import to_discrete_single, to_discrete_double


def build_blockchain(ohlcv, **kwargs):
	if ohlcv is None:
		raise RuntimeError('No ohlcv loaded!')
	if kwargs.get('blockchain') is None:
		raise RuntimeError('No ohlcv loaded!')

	return