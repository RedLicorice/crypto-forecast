from lib.dataset import DatasetFactory
import pandas as pd
from collections import deque

def build(ohlcv, **kwargs):
	_cols = ['open', 'high', 'low', 'close', 'volume']
	_tf = kwargs.get('periods', 7)
	q = deque([], _tf)
	result = []
	# There must be a better way to do this
	for i, row in ohlcv.iterrows():
		q.append(row)  # Add to the right side of the queue
		# Build a snapshot of current situation from the queue
		r = {
			'open': q[0]['open'],  # First element's open
			'close': q[-1]['close'],
			'high': max([e['high'] for e in q]),
			'low': min([e['low'] for e in q]),
			'volume': sum([e['volume'] for e in q])
		}
		result.append(r)
	return pd.DataFrame(data=result, index=ohlcv.index)

def reverse_resample(df, **kwargs):
	period = kwargs.get('period', 7)
	_period = kwargs.get('type', 'D')
	return df.resample('{}{}'.format(period, _period), closed='left', label='right', convention='end', kind='timestamp', loffset='-1D') \
		.agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})

def period_resample(df, **kwargs):
	period = int(kwargs.get('period', 7))
	result = []
	df = df.sort_index().copy()
	for i in range(period):
		_df = df.iloc[i:]
		nth_day = _df.resample('{}D'.format(period), closed='left', label='right', convention='end', kind='timestamp', loffset='-1D') \
				.agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).copy()
		result.append(nth_day)
	return pd.concat(result, sort=True).sort_index()

DatasetFactory.register_features('timeframe', build)
DatasetFactory.register_features('reverse_resample', reverse_resample)
DatasetFactory.register_features('period_resample', period_resample)