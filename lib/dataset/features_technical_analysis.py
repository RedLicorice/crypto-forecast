import pandas as pd
from lib.utils import to_discrete_single, to_discrete_double
from lib.technical_indicators import *

TA_DEFAULT_INDICATORS = {
	'rsma' : [(5,20), (8,15), (20,50)],
	'rema' : [(5,20), (8,15), (20,50)],
	'macd' : [(12,26)],
	'ao' : [14],
	'adx' : [14],
	'wd' : [14],
	'ppo' : [(12,26)],
	'rsi':[14],
	'mfi':[14],
	'tsi':None,
	'stoch':[14],
	'cmo':[14],
	'atrp':[14],
	'pvo':[(12,26)],
	'fi':[13,50],
	'adi':None,
	'obv':None
}

def features_ta(ohlcv, **kwargs):
	if ohlcv is None:
		raise RuntimeError('No ohlcv loaded!')
	mode = kwargs.get('mode', 'continuous')

	ta = get_ta_features(
		ohlcv['high'].values,
		ohlcv['low'].values,
		ohlcv['close'].values,
		ohlcv['volume'].values,
		kwargs.get('indicators', TA_DEFAULT_INDICATORS)
	)
	if mode == 'discrete':
		ta = discretize_ta_features(ta)

	result = pd.DataFrame(index=ohlcv.index)
	for k in ta.keys():  # Keys are the same both for 'ta' and 'dta'
		result[k] = ta[k]

	if mode == 'variation':
		for k in ta.keys():
			result[k] = result[k].pct_change(periods=kwargs.get('periods',1))

	return result

def get_ta_features(high, low, close, volume, desc):
	"""
	Returns a dict containing the technical analysis indicators calculated on the given
	high, low, close and volumes.
	"""
	ta = {}

	# Set numpy to ignore division error and invalid values (since not all features are complete)
	old_settings = np.seterr(divide='ignore', invalid='ignore')

	# Determine relative moving averages
	for _short, _long in desc['rsma']:
		ta['rsma_{}_{}'.format(_short, _long)] = relative_sma(close, _short, _long)
	for _short, _long in desc['rema']:
		ta['rema_{}_{}'.format(_short, _long)] = relative_ema(close, _short, _long)

	# MACD Indicator
	if 'macd' in desc:
		for _short, _long in desc['macd']:
			ta['macd_{}_{}'.format(_short, _long)] = moving_average_convergence_divergence(close, _short, _long)

	# Aroon Indicator
	if 'ao' in desc:
		for _period in desc['ao']:
			ta['ao_{}'.format(_period)] = aroon_oscillator(close, _period)

	# Average Directional Movement Index (ADX)
	if 'adx' in desc:
		for _period in desc['adx']:
			ta['adx_'.format(_period)] = average_directional_index(close, high, low, _period)

	# Difference between Positive Directional Index(DI+) and Negative Directional Index(DI-)
	if 'wd' in desc:
		for _period in desc['wd']:
			ta['wd_{}'.format(_period)] = \
				positive_directional_index(close, high, low, _period) \
				- negative_directional_index(close, high, low, _period)

	# Percentage Price Oscillator
	if 'ppo' in desc:
		for _short, _long in desc['ppo']:
			ta['ppo_{}_{}'] = price_oscillator(close, _short, _long)

	# Relative Strength Index
	if 'rsi' in desc:
		for _period in desc['rsi']:
			ta['rsi_{}'.format(_period)] = relative_strength_index(close, _period)

	# Money Flow Index
	if 'mfi' in desc:
		for _period in desc['mfi']:
			ta['mfi'.format(_period)] = money_flow_index(close, high, low, volume, _period)

	# True Strength Index
	if 'tsi' in desc:
		ta['tsi'] = true_strength_index(close)

	# Stochastic Oscillator
	if 'stoch' in desc:
		for _period in desc['stoch']:
			ta['stoch_{}'.format(_period)] = percent_k(close, _period)
	# ta.py['stoch'] = percent_k(high, low, close, 14)

	# Chande Momentum Oscillator
	## Not available in ta.py
	if 'cmo' in desc:
		for _period in desc['cmo']:
			ta['cmo_{}'.format(_period)] = chande_momentum_oscillator(close, _period)

	# Average True Range Percentage
	if 'atrp' in desc:
		for _period in desc['atrp']:
			ta['atrp_{}'.format(_period)] = average_true_range_percent(close, _period)

	# Percentage Volume Oscillator
	if 'pvo' in desc:
		for _short, _long in desc['pvo']:
			ta['pvo_{}_{}'.format(_short, _long)] = volume_oscillator(volume, _short, _long)

	# Force Index
	if 'fi' in desc:
		fi = force_index(close, volume)
		for _period in desc['fi']:
			ta['fi_{}'.format(_period)] = exponential_moving_average(fi, _period)

	# Accumulation Distribution Line
	if 'adi' in desc:
		ta['adi'] = accumulation_distribution(close, high, low, volume)

	# On Balance Volume
	if 'obv' in desc:
		ta['obv'] = on_balance_volume(close, volume)

	# Restore numpy error settings
	np.seterr(**old_settings)

	return ta

def discretize_ta_features(ta):
	"""
	Returns a dict containing the discretized version of the input
	dict of technical indicators
	"""
	dta = {}
	for k,v in ta.items():
		if k.startswith(['rsma', 'rema', 'macd', 'pvo', 'fi', 'adi', 'obv', 'ao', 'wd', 'ppo']):
			dta[k] = to_discrete_single(v, 0)
		elif k.startswith('adx'):
			dta[k] = to_discrete_single(v, 20)
		elif k.startswith('rsi'):
			dta[k] = to_discrete_double(v, 30, 70)
		elif k.startswith('mfi'):
			dta[k] = to_discrete_double(v, 30, 70)
		elif k.startswith('tsi'):
			dta[k] = to_discrete_double(v, -25, 25)
		elif k.startswith('stoch'):
			dta[k] = to_discrete_double(v, 20, 80)
		elif k.startswith('cmo'):
			dta[k] = to_discrete_double(v, -50, 50)
		elif k.startswith('atrp'):
			dta[k] = to_discrete_single(v, 30)
	for k in dta.keys():
		dta[k] = [np.nan if np.isnan(x) else np.asscalar(x) for x in dta[k]]
	return dta

