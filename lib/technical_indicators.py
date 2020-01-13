import numpy as np
from pyti.exponential_moving_average import exponential_moving_average
from pyti.simple_moving_average import simple_moving_average
from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence
from pyti.aroon import aroon_up, aroon_down, aroon_oscillator
from pyti.directional_indicators import (average_directional_index, positive_directional_index, negative_directional_index)
from pyti.price_oscillator import price_oscillator
from pyti.relative_strength_index import relative_strength_index
from pyti.money_flow_index import money_flow_index
from pyti.stochastic import percent_k
from pyti.chande_momentum_oscillator import chande_momentum_oscillator
from pyti.average_true_range_percent import average_true_range_percent
from pyti.volume_oscillator import volume_oscillator
from pyti.accumulation_distribution import accumulation_distribution
from pyti.on_balance_volume import on_balance_volume
from pyti.force_index import force_index
from pyti.true_strength_index import true_strength_index

from pyti.function_helper import fill_for_noncomputable_vals
from pyti import catch_errors
import warnings

# def simple_moving_average(data, period):
# 	"""
# 	Simple Moving Average.
#
# 	Formula:
# 	SUM(data / N)
# 	"""
# 	catch_errors.check_for_period_error(data, period)
# 	# Mean of Empty Slice RuntimeWarning doesn't affect output so it is
# 	# supressed
# 	with warnings.catch_warnings():
# 		warnings.simplefilter("ignore", category=RuntimeWarning)
# 		ret = np.cumsum(data, dtype=float)
# 		ret[period:] = ret[period:] - ret[:-period]
# 		sma= ret[period - 1:] / period
# 	sma = fill_for_noncomputable_vals(data, sma)
# 	return sma


def relative_sma(data, short, long):
		sma_short = simple_moving_average(data, period=short)
		sma_long = simple_moving_average(data, period=long)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=RuntimeWarning)
			smadiff = sma_short - sma_long
			rsma = np.divide(smadiff, sma_long)
		return fill_for_noncomputable_vals(data, rsma)


def relative_ema(data, short, long):
	ema_short = exponential_moving_average(data, period=short)
	ema_long = exponential_moving_average(data, period=long)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=RuntimeWarning)
		emadiff = ema_short - ema_long
		rema = np.divide(emadiff, ema_long)
	return fill_for_noncomputable_vals(data, rema)

# def accumulation_distribution_mine(close_data, high_data, low_data, volume):
# 	"""
# 	Accumulation/Distribution.
#
# 	Formula:
# 	A/D = (Ct - Lt) - (Ht - Ct) / (Ht - Lt) * Vt + A/Dt-1
# 	"""
# 	catch_errors.check_for_input_len_diff(
# 		close_data, high_data, low_data, volume
# 	)
#
# 	with warnings.catch_warnings():
# 		warnings.simplefilter("ignore", category=RuntimeWarning)
#
# 		ct_lt = close_data - low_data
# 		ht_ct = high_data - close_data
# 		ht_lt = high_data - low_data
# 		ct_lt_ht_ct= ct_lt - ht_ct
# 		mfm = ct_lt_ht_ct / ht_lt # Money Flow Multiplier
# 		mfv = mfm * volume # Money Flow Volume
#
# 	ad = np.zeros(len(close_data))
# 	for idx in range(1, len(mfv)):
# 		ad[idx] = mfv[idx-1] + ad[idx-1]
# 	return ad
#
# def accumulation_distribution(close_data, high_data, low_data, volume, **kwargs):
# 	"""
# 	Accumulation/Distribution.
# 	Formula:
# 	A/D = (Ct - Lt) - (Ht - Ct) / (Ht - Lt) * Vt + A/Dt-1
# 	"""
# 	catch_errors.check_for_input_len_diff(
# 		close_data, high_data, low_data, volume
# 		)
#
# 	ad = np.zeros(len(close_data))
# 	for idx in range(1, len(close_data)):
# 		candle = high_data[idx] - low_data[idx]
# 		if candle == 0:
# 			if high_data[idx] != close_data[idx]:
# 				if kwargs.get("ignore_errors", False):
# 					ad[idx] = ad[idx - 1] # Smooth
# 				else:
# 					raise RuntimeError("High and low are equals but close is not.")
# 			else:
# 				ad[idx] = ad[idx - 1]
# 		else:
# 			ad[idx] = (
# 				(((close_data[idx] - low_data[idx]) -
# 					(high_data[idx] - close_data[idx])) /
# 					(high_data[idx] - low_data[idx]) *
# 					volume[idx]) +
# 				ad[idx-1]
# 				)
# 	return ad
#
def percent_k_pr(high_data, low_data, close_data, period):
	"""
	%K.
	Formula:
	%k = data(t) - low(n) / (high(n) - low(n))
	"""
	# print (len(high_data))
	# print (period)
	catch_errors.check_for_period_error(high_data, period)
	catch_errors.check_for_period_error(low_data, period)
	catch_errors.check_for_period_error(close_data, period)
	percent_k = [((close_data[idx] - np.min(low_data[idx+1-period:idx+1])) /
		 (np.max(high_data[idx+1-period:idx+1]) -
		  np.min(low_data[idx+1-period:idx+1]))) for idx in range(period-1, len(close_data))]
	percent_k = fill_for_noncomputable_vals(close_data, percent_k)

	return percent_k
#
# def aroon_oscillator(data, period):
# 	"""
# 	Aroon Oscillator.
# 	Formula:
# 	AO = AROON_UP(PERIOD) - AROON_DOWN(PERIOD)
# 	"""
# 	catch_errors.check_for_period_error(data, period)
# 	period = int(period)
# 	return aroon_up(data, period) - aroon_down(data, period)
#
# def relative_strength_index(data, period):
# 	"""
# 	Relative Strength Index.
# 	Formula:
# 	RSI = 100 - (100 / 1 + (prevGain/prevLoss))
# 	"""
# 	catch_errors.check_for_period_error(data, period)
#
# 	period = int(period)
# 	changes = [data_tup[1] - data_tup[0] for data_tup in zip(data[::1], data[1::1])]
#
# 	gains = [0 if val < 0 else val for val in changes]
# 	losses = [0 if val > 0 else abs(val) for val in changes]
#
# 	avg_gain = np.mean(gains[:period])
# 	avg_loss = np.mean(losses[:period])
#
# 	rsi = []
# 	if avg_loss == 0:
# 		rsi.append(100)
# 	else:
# 		rs = avg_gain / avg_loss
# 		rsi.append(100 - (100 / (1 + rs)))
#
# 	for idx in range(1, len(data) - period):
# 		avg_gain = ((avg_gain * (period - 1) +
# 					gains[idx + (period - 1)]) / period)
# 		avg_loss = ((avg_loss * (period - 1) +
# 					losses[idx + (period - 1)]) / period)
#
# 		if avg_loss == 0:
# 			rsi.append(100)
# 		else:
# 			rs = avg_gain / avg_loss
# 			rsi.append(100 - (100 / (1 + rs)))
#
# 	rsi = fill_for_noncomputable_vals(data, rsi)
#
# 	return rsi
#
# def true_strength_index(close_data):
# 	"""
# 	True Strength Index.
# 	Double Smoothed PC
# 	------------------
# 	PC = Current Price minus Prior Price
# 	First Smoothing = 25-period EMA of PC
# 	Second Smoothing = 13-period EMA of 25-period EMA of PC
# 	Double Smoothed Absolute PC
# 	---------------------------
# 	Absolute Price Change |PC| = Absolute Value of Current Price minus Prior Price
# 	First Smoothing = 25-period EMA of |PC|
# 	Second Smoothing = 13-period EMA of 25-period EMA of |PC|
# 	TSI = 100 x (Double Smoothed PC / Double Smoothed Absolute PC)
# 	"""
# 	if len(close_data) < 40:
# 		raise RuntimeError("Data must have at least 40 items")
#
# 	pc = np.diff(close_data, 1)
# 	apc = np.abs(pc)
#
# 	num = exponential_moving_average(pc, 25)
# 	num = exponential_moving_average(num, 13)
# 	den = exponential_moving_average(apc, 25)
# 	den = exponential_moving_average(den, 13)
#
# 	tsi = 100 * num / den
# 	tsi = fill_for_noncomputable_vals(close_data, tsi)
# 	return tsi