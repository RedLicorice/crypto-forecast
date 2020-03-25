import pandas as pd
from collections import deque
from lib.dataset import features_ta, get_ta_config

def features_timeframe(ohlcv, **kwargs):
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
	rf = df.resample('{}{}'.format(period, _period), closed='left', label='right', convention='end', kind='timestamp', loffset='-1D') \
		.agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})

def periodic_ohlcv_resample(ohlcv, **kwargs):
	period = int(kwargs.get('period', 7))
	result = []
	df = ohlcv.sort_index().copy()
	for i in range(period):
		_df = df.iloc[i:]
		nth_day = _df.resample('{}D'.format(period), closed='left', label='right', convention='end', kind='timestamp', loffset='-1D') \
				.agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).copy()
		if kwargs.get('label'):
			nth_day.columns = ['{}_{}'.format(c, period) for c in nth_day.columns]
		result.append(nth_day)
	_result = pd.concat(result, sort=True).sort_index()
	_result = _result.loc[ohlcv.first_valid_index():ohlcv.last_valid_index()]
	return _result

def periodic_ohlcv_pct_change(ohlcv, **kwargs):
	period = int(kwargs.get('period', 7))
	pct_period = int(kwargs.get('pct_period', 1))
	result = []
	df = ohlcv.sort_index().copy()
	for i in range(period):
		_df = df.iloc[i:]
		nth_day = _df.resample('{}D'.format(period), closed='left', label='right', convention='end', kind='timestamp', loffset='-1D') \
				.agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})\
				.pct_change(pct_period)\
				.copy()
		nth_day.columns = ['{}_pct{}_{}'.format(c, pct_period, period) for c in nth_day.columns]
		result.append(nth_day)
	_result = pd.concat(result, sort=True).sort_index()
	_result = _result.loc[ohlcv.first_valid_index():ohlcv.last_valid_index()]
	return _result

def period_resampled_ta(ohlcv, **kwargs):
	period = int(kwargs.get('period', 7))
	result = []
	df = ohlcv.sort_index().copy()
	_ind = get_ta_config(period)
	# Fai un resample per ciascuno dei primi N giorni
	for i in range(period):
		_df = df.iloc[i:]
		nth_day = _df.resample('{}D'.format(period), closed='left', label='right', convention='end', kind='timestamp', loffset='-1D') \
				.agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
		nth_day_ta = features_ta(nth_day, indicators=_ind, mode=kwargs.get('mode', 'continuous'))
		result.append(nth_day_ta)
	_result = pd.concat(result, sort=True).sort_index()
	_result.columns = ['{}_{}'.format(c, period) for c in _result.columns]
	_result = _result.loc[ohlcv.first_valid_index():ohlcv.last_valid_index()]
	return _result