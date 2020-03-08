from lib.dataset import DatasetFactory
import pandas as pd
from lib.utils import to_discrete_single, to_discrete_double
from lib.technical_indicators import *

def build(ohlcv, **kwargs):
	if ohlcv is None:
		raise RuntimeError('No ohlcv loaded!')
	mode = kwargs.get('mode', 'continuous')

	ta = get_ta_features(ohlcv['high'].values,
							  ohlcv['low'].values,
							  ohlcv['close'].values,
							  ohlcv['volume'].values)
	if mode == 'discrete':
		ta = discretize_ta_features(ta)

	result = pd.DataFrame(index=ohlcv.index)
	for k in ta.keys():  # Keys are the same both for 'ta' and 'dta'
		result[k] = ta[k]

	if mode == 'variation':
		for k in ta.keys():
			result[k] = result[k].pct_change(periods=kwargs.get('periods',1))

	return result

def get_ta_features(high, low, close, volume):
	"""
	Returns a dict containing the technical analysis indicators calculated on the given
	high, low, close and volumes.
	"""
	ta = {}

	# Set numpy to ignore division error and invalid values (since not all features are complete)
	old_settings = np.seterr(divide='ignore', invalid='ignore')

	# Determine relative moving averages
	ta['rsma5_20'] = relative_sma(close, 5, 20)
	ta['rsma8_15'] = relative_sma(close, 8, 15)
	ta['rsma20_50'] = relative_sma(close, 20, 50)
	ta['rema5_20'] = relative_ema(close, 5, 20)
	ta['rema8_15'] = relative_ema(close, 8, 15)
	ta['rema20_50'] = relative_ema(close, 20, 50)

	# MACD Indicator
	ta['macd_12_26'] = moving_average_convergence_divergence(close, 12, 26)

	# Aroon Indicator
	ta['ao'] = aroon_oscillator(close, 14)

	# Average Directional Movement Index (ADX)
	ta['adx'] = average_directional_index(close, high, low,
										  14)

	# Difference between Positive Directional Index(DI+) and Negative Directional Index(DI-)
	ta['wd'] = \
		positive_directional_index(close, high, low, 14) \
		- negative_directional_index(close, high, low, 14)

	# Percentage Price Oscillator
	ta['ppo'] = price_oscillator(close, 12, 26)

	# Relative Strength Index
	ta['rsi'] = relative_strength_index(close, 14)

	# Money Flow Index
	ta['mfi'] = money_flow_index(close, high, low, volume, 14)

	# True Strength Index
	ta['tsi'] = true_strength_index(close)

	# Stochastic Oscillator
	ta['stoch'] = percent_k(close, 14)
	# ta.py['stoch'] = percent_k(high, low, close, 14)

	# Chande Momentum Oscillator
	## Not available in ta.py
	ta['cmo'] = chande_momentum_oscillator(close, 14)

	# Average True Range Percentage
	ta['atrp'] = average_true_range_percent(close, 14)

	# Percentage Volume Oscillator
	ta['pvo'] = volume_oscillator(volume, 12, 26)

	# Force Index
	fi = force_index(close, volume)
	ta['fi13'] = exponential_moving_average(fi, 13)
	ta['fi50'] = exponential_moving_average(fi, 50)

	# Accumulation Distribution Line
	ta['adi'] = accumulation_distribution(close, high, low,
										  volume)

	# On Balance Volume
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
	dta['rsma5_20'] = to_discrete_single(ta['rsma5_20'], 0)
	dta['rsma8_15'] = to_discrete_single(ta['rsma8_15'], 0)
	dta['rsma20_50'] = to_discrete_single(ta['rsma20_50'], 0)
	dta['rema5_20'] = to_discrete_single(ta['rema5_20'], 0)
	dta['rema8_15'] = to_discrete_single(ta['rema8_15'], 0)
	dta['rema20_50'] = to_discrete_single(ta['rema20_50'], 0)
	dta['macd_12_26'] = to_discrete_single(ta['macd_12_26'], 0)
	dta['ao'] = to_discrete_single(ta['ao'], 0)
	dta['adx'] = to_discrete_single(ta['adx'], 20)
	dta['wd'] = to_discrete_single(ta['wd'], 0)
	dta['ppo'] = to_discrete_single(ta['ppo'], 0)
	dta['rsi'] = to_discrete_double(ta['rsi'], 30, 70)
	dta['mfi'] = to_discrete_double(ta['mfi'], 30, 70)
	dta['tsi'] = to_discrete_double(ta['tsi'], -25, 25)
	dta['stoch'] = to_discrete_double(ta['stoch'], 20, 80)
	dta['cmo'] = to_discrete_double(ta['cmo'], -50, 50)
	dta['atrp'] = to_discrete_single(ta['atrp'], 30)
	dta['pvo'] = to_discrete_single(ta['pvo'], 0)
	dta['fi13'] = to_discrete_single(ta['fi13'], 0)
	dta['fi50'] = to_discrete_single(ta['fi50'], 0)
	dta['adi'] = to_discrete_single(ta['adi'], 0)
	dta['obv'] = to_discrete_single(ta['obv'], 0)
	for k in dta.keys():
		dta[k] = [np.nan if np.isnan(x) else np.asscalar(x) for x in dta[k]]
	return dta

DatasetFactory.register_features('technical_analysis', build)
