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