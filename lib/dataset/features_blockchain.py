from lib.dataset import DatasetFactory
import pandas as pd
from lib.utils import to_discrete_single, to_discrete_double


def build(ohlcv, **kwargs):
	if ohlcv is None:
		raise RuntimeError('No ohlcv loaded!')

	return

DatasetFactory.register_features('blockchain', build)