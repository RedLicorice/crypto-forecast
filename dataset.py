import pandas as pd
from functools import reduce
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from technical_indicators import *
from math import isnan
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from keras.utils import to_categorical

from plotter import correlation

import json

class DatasetBuilder:
	def __init__(self):
		pass

	## Data loading

	def make_ohlcv(self, symbol, ohlc, vol):
		ohlcv = ohlc.join(vol[[symbol+'_Volume']], how='left', sort=True)
		# Drop all columns except the ones related to this symbol
		keep = ['Date']
		for col in ohlcv.columns:
			if symbol in col:
				keep.append(col)
		ohlcv.drop(ohlcv.columns.difference(keep), 1, inplace=True)
		return ohlcv

	def change_ohlcv_symbol(self, ohlcv, mapping):
		for col_name, symbol in mapping.items():
			_map = {
				col_name : symbol,
				col_name + '_Open': symbol + '_Open',
				col_name + '_High': symbol + '_High',
				col_name + '_Low' : symbol + '_Low',
				col_name + '_Volume' : symbol + '_Volume'
			}
			ohlcv.rename(columns = _map, inplace=True)
		return ohlcv

	def add_blockchain_data(self, input, chain, symbol):
		# Rename index for join
		map = {'date': 'Date'}
		for col in chain.columns:
			if str(col) == symbol or str(col) == 'date':
				continue
			map[str(col)] = '{symbol}_{col}'.format(symbol=symbol, col=str(col))
		chain.rename(columns=map, inplace=True)
		return input.join(chain, how='left', sort=True)

	def add_ta_features(self, df, symbol, discrete=False):
		# Set numpy to ignore division error and invalid values (since not all datasets are complete)
		old_settings = np.seterr(divide='ignore',invalid='ignore')

		col_close = symbol
		col_open = symbol + '_Open'
		col_high = symbol + '_High'
		col_low = symbol + '_Low'
		col_volume = symbol + '_Volume'


		# Determine moving averages
		#for n in [5, 8, 15, 20, 50]:
		#	df[symbol+'_sma_'+str(n)] = simple_moving_average(df[col_close].values, n)
		#	df[symbol+'_ema_'+str(n)] = exponential_moving_average(df[col_close].values, n)

		# Determine relative moving averages
		df[symbol+'_rsma5_20'] = relative_sma(df[col_close].values, 5, 20)
		df[symbol+'_rsma8_15'] = relative_sma(df[col_close].values, 8, 15)
		df[symbol+'_rsma20_50'] = relative_sma(df[col_close].values, 20,50)
		df[symbol+'_rema5_20'] = relative_ema(df[col_close].values, 5, 20)
		df[symbol+'_rema8_15'] = relative_ema(df[col_close].values, 8, 15)
		df[symbol+'_rema20_50'] = relative_ema(df[col_close].values, 20, 50)

		# MACD Indicator
		df[symbol+'_macd_12_26'] = moving_average_convergence_divergence(df[col_close].values, 12, 26)

		# Aroon Indicator
		df[symbol+'_ao'] = aroon_oscillator(df[col_close].values,14)

		# Average Directional Movement Index (ADX)
		df[symbol+'_adx'] = average_directional_index(df[col_close].values, df[col_high].values, df[col_low].values, 14)

		# Difference between Positive Directional Index(DI+) and Negative Directional Index(DI-)
		df[symbol+'_wd'] = \
			positive_directional_index(df[col_close].values, df[col_high].values, df[col_low].values, 14)\
			- negative_directional_index(df[col_close].values, df[col_high].values, df[col_low].values, 14)

		# Percentage Price Oscillator
		df[symbol+'_ppo'] = price_oscillator(df[col_close].values, 12, 26)

		# Relative Strength Index
		df[symbol+'_rsi'] = relative_strength_index(df[col_close].values, 14)

		# Money Flow Index
		df[symbol+'_mfi'] = money_flow_index(df[col_close].values,  df[col_high].values, df[col_low].values, df[col_volume].values, 14)

		# True Strength Index
		df[symbol+'_tsi'] = true_strength_index(df[col_close].values)

		# Stochastic Oscillator
		df[symbol+'_stoch'] = percent_k(df[col_close].values, 14)
		#df[symbol+'_stoch'] = percent_k(df[col_high].values, df[col_low].values, df[col_close].values, 14)

		# Chande Momentum Oscillator
		## Not available in ta
		df[symbol+'_cmo'] = chande_momentum_oscillator(df[col_close].values, 14)

		# Average True Range Percentage
		df[symbol+'_atrp'] = average_true_range_percent(df[col_close].values, 14)

		# Percentage Volume Oscillator
		df[symbol+'_pvo'] = volume_oscillator(df[col_volume].values, 12, 26)

		# Force Index
		fi = force_index(df[col_close].values, df[col_volume].values)
		df[symbol+'_fi13'] = exponential_moving_average(fi, 13)
		df[symbol+'_fi50'] = exponential_moving_average(fi, 50)

		# Accumulation Distribution Line
		df[symbol+'_adi'] = accumulation_distribution(df[col_close].values,  df[col_high].values, df[col_low].values,
													  df[col_volume].values)

		# On Balance Volume
		df[symbol+'_obv'] = on_balance_volume(df[col_close].values, df[col_volume].values)

		# Restore numpy error settings
		np.seterr(**old_settings)
		if discrete:
			df[symbol + '_rsma5_20'] = self.to_discrete_single(df[symbol + '_rsma5_20'], 0)
			df[symbol + '_rsma8_15'] = self.to_discrete_single(df[symbol + '_rsma8_15'], 0)
			df[symbol + '_rsma20_50'] = self.to_discrete_single(df[symbol + '_rsma20_50'], 0)
			df[symbol + '_rema5_20'] = self.to_discrete_single(df[symbol + '_rema5_20'], 0)
			df[symbol + '_rema8_15']  = self.to_discrete_single(df[symbol + '_rema8_15'], 0)
			df[symbol + '_rema20_50'] = self.to_discrete_single(df[symbol + '_rema20_50'], 0)
			df[symbol + '_macd_12_26'] = self.to_discrete_single(df[symbol + '_macd_12_26'], 0)
			df[symbol + '_ao'] = self.to_discrete_single(df[symbol + '_ao'], 0)
			df[symbol + '_adx'] = self.to_discrete_single(df[symbol + '_adx'], 20)
			df[symbol + '_wd'] = self.to_discrete_single(df[symbol + '_wd'], 0)
			df[symbol + '_ppo'] = self.to_discrete_single(df[symbol + '_ppo'], 0)
			df[symbol + '_rsi'] = self.to_discrete_double(df[symbol + '_rsi'], 30, 70)
			df[symbol + '_mfi'] = self.to_discrete_double(df[symbol + '_mfi'], 30, 70)
			df[symbol + '_tsi'] = self.to_discrete_double(df[symbol + '_tsi'], -25, 25)
			df[symbol + '_stoch'] = self.to_discrete_double(df[symbol + '_stoch'], 20, 80)
			df[symbol + '_cmo'] = self.to_discrete_double(df[symbol + '_cmo'], -50, 50)
			df[symbol + '_atrp'] = self.to_discrete_single(df[symbol + '_atrp'], 30)
			df[symbol + '_pvo'] = self.to_discrete_single(df[symbol + '_pvo'], 0)
			df[symbol + '_fi13'] = self.to_discrete_single(df[symbol + '_fi13'], 0)
			df[symbol + '_fi50'] = self.to_discrete_single(df[symbol + '_fi50'], 0)
			df[symbol + '_adi'] = self.to_discrete_single(df[symbol + '_adi'], 0)
			df[symbol + '_obv'] = self.to_discrete_single(df[symbol + '_obv'], 0)
		return df

	## Operations on columns
	def fill_holes(self, values, hole=np.nan):
		imp = SimpleImputer(missing_values=hole, strategy='mean')
		old_shape = values.shape
		resh = np.reshape(values, (-1,1))
		imp.fit(resh)
		res = imp.transform(resh)
		res = np.reshape(res, old_shape)
		return res

	# Discretization
	def to_discrete_single(self, values, threshold):
		def _to_discrete(x, threshold):
			if np.isnan(x):
				return np.nan
			if x < threshold:
				return 1
			return 2
		fun = np.vectorize(_to_discrete)
		return fun(values, threshold)

	def to_discrete_double(self, values, threshold_lo = 0.01, threshold_hi = 0.01):
		def _to_discrete(x, threshold_lo, threshold_hi):
			if np.isnan(x):
				return np.nan
			if x <= threshold_lo:
				return 1
			elif threshold_lo < x < threshold_hi:
				return 2
			else:
				return 3
		fun = np.vectorize(_to_discrete)
		return fun(values, threshold_lo, threshold_hi)

	def discrete_label(self, values):
		def _to_label(cls):
			if np.isnan(cls):
				return np.nan
			if cls == 1:
				return 'Decrease'
			elif cls == 2:
				return 'Stationary'
			elif cls == 3:
				return 'Increase'
		return np.vectorize(_to_label)(values)

	# Analysis
	def lda_reduction(self, X, y):
		lda = LinearDiscriminantAnalysis(n_components=12)
		X_lda = lda.fit(X, y).transform(X)
		# Print the number of features
		print('Original number of features:', X.shape[1])
		print('Reduced number of features:', X_lda.shape[1])
		for x,var in zip(X_lda, lda.explained_variance_ratio_):
			print('{} = {}'.format(x,var))
		print(X_lda)
		return X_lda

	def adf_test(self, timeseries, significance=.05, printResults=True):
		# Dickey-Fuller test:
		adfstat, pvalue, usedlag, nobs, critvalues, icbest = adfuller(timeseries, autolag='AIC')


		if (pvalue < significance):
			self.isStationary = True
		else:
			self.isStationary = False

		if printResults:
			# Add Critical Values

			print('Augmented Dickey-Fuller Test Results:')
			print('ADF Test Statistic = %.8f' % adfstat)
			print('P-Value = %.8f' % pvalue)
			print('# Lags Used = %d' % usedlag)
			print('# Observations Used = %d' % nobs)
			for key, value in critvalues.items():
				print('Critical Value ({}={})'.format(key,value))
		crit = {}
		reject_by_adf = 0
		for k,v in critvalues.items():
			crit[k] = {'value':v,'null_hp_reject_by_adfstat':'True (Stationary)' if adfstat <= v else 'False (Non-Stationary)'}
			if adfstat <= v:
				reject_by_adf = 1
		reject_by_pvalue = pvalue <= significance
		reject = reject_by_adf and reject_by_pvalue
		potential_reject = reject_by_adf ^ reject_by_pvalue # if both are true then it's reject already
		return reject, potential_reject, {
			'adfstat':adfstat,
			'pvalue':pvalue,
			'null_hp_reject_by_pvalue': 'True (Stationary)' if pvalue <= significance else 'False (Non Stationary)',
			'reject_by_pvalue': 1 if reject_by_pvalue else 0,
			'reject_by_adf': reject_by_adf,
			'reject': 1 if reject else 0,
			'usedlag':usedlag,
			'nobs':nobs,
			'critvalues': crit,
		}

	# Checks
	def check_dataset(self, df):
		for col in df.columns:
			self.check_integrity(df, col)

	def check_integrity(self, df, col):
		for i,v in enumerate(df[col].values):
			if not v:
				print("{} contains zeroes at index {} - {}".format(col, i, df.index[i]))
			if v == None:
				print("{} contains None at index {} - {}".format(col, i, df.index[i]))
			if isnan(v):
				print("{} contains NaN at index {} - {}".format(col, i, df.index[i]))

	def get_non_numeric(self, df):
		not_numeric = []
		for col in df.columns:
			if not np.issubdtype(df[col].dtype, np.number):
				not_numeric.append(col)
		return not_numeric


# Testing, this is NOT a driver
if __name__ == '__main__':
	#np.seterr(all='raise')
	db = DatasetBuilder()
	# Join OHLC and Volume datasets for each year
	datasets = []
	for year in [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]:
		print('Year: '+str(year))
		ohlc = pd.read_csv('data/polito/'+str(year)+'_candle.csv', sep=',', index_col='Date', parse_dates=True)
		vol = pd.read_csv('data/polito/'+str(year)+'_volume.csv', sep=',', index_col='Date', parse_dates=True)
		ohlcv = db.make_ohlcv('BTC', ohlc, vol)
		datasets.append(ohlcv)
	# Merge yearly datasets
	ohlcv = reduce(lambda left, right: left.append(right), datasets)
	print("OHLCV rows: " + str(ohlcv.shape[0]))

	# Build the dataset
	dataset = db.add_ta_features(ohlcv, 'BTC', discrete=True)
	print("OHLCV+TA rows: " + str(dataset.shape[0]))

	# Add blockchain information
	# chain = pd.read_csv('data/coinmetrics.io/btc.csv', sep=',', index_col='date', parse_dates=True)
	# chain.drop(columns=['PriceUSD', 'PriceBTC'], inplace=True) # Drop PriceUSD because it's redundant
	# chain.dropna(axis=0, how='all', inplace=True) # Drop columns composed of NaN values
	# clean_chain = chain.replace([np.inf, -np.inf], np.nan).interpolate(method='linear', axis=1)
	# dataset = db.add_blockchain_data(dataset, clean_chain, 'BTC')
	# print("OHLCV+TA+Blockchain rows: " + str(dataset.shape[0]))

	# Add Y
	y_ref = dataset[['BTC']].copy().values
	# Variation w.r.t. next period
	# calculated as variation w.r.t previous period, shifted backwards by 1.
	# NaN's are filled by last valid value.
	y_var = np.roll(dataset['BTC'].pct_change(1, fill_method='ffill').fillna(0), -1)
	y = db.to_discrete_double(y_var, -0.01, 0.01)
	y_label = db.discrete_label(y)

	# Make data stationary by differencing
	#for col in ['BTC','BTC_High','BTC_Low','BTC_Open','BTC_Volume']:
		#diff = dataset[col].diff(periods=1).fillna(0)
		#dataset[col] = diff

	# Make data discrete
	discretization_refs = {}
	for col in ['BTC', 'BTC_High', 'BTC_Low', 'BTC_Open', 'BTC_Volume']:
		# variation w.r.t previous period
		c_var = dataset[col].pct_change(1, fill_method='ffill').fillna(0)
		c = db.to_discrete_double(c_var, -0.01, 0.01)
		discretization_refs[col + '_ref'] = dataset[col].copy()
		discretization_refs[col + '_var'] = c_var
		discretization_refs[col + '_class'] = c
		discretization_refs[col + '_label'] = db.discrete_label(c_var)
		dataset[col] = c
		db.check_integrity(dataset, col)

	# Fill holes linearly [Breaks categorical]
	#dataset = dataset.replace([np.inf, -np.inf], np.nan).interpolate(method='linear', axis=1)

	# Normalize data
	#scaler = StandardScaler()
	#for col in ['BTC','BTC_Open','BTC_High','BTC_Low','BTC_Volume']:
	#	reshaped = np.reshape(dataset[col].values, (-1,1))
	#	dataset[col] = scaler.fit_transform(reshaped)

	# Append target
	dataset['y'] = y

	# Save dataset and correlation matrix
	dataset.to_csv('data/result/dataset.csv', sep=',', encoding='utf-8', index=True, index_label='Date')
	corr = dataset.corr()
	corr.to_csv('data/result/dataset_corr.csv', sep=',', encoding='utf-8', index=True, index_label='index')
	correlation(corr, 'data/result/dataset_corr.png')

	# Print some rows to check everything is alright
	pd.set_option('display.float_format', lambda x: '%.12f' % x)
	checkset = pd.DataFrame(index=dataset.index)
	for name, series in discretization_refs.items():
		checkset[name] = series
	checkset['close'] = y_ref
	checkset['y'] = y
	checkset['y_var'] = y_var
	checkset['y_label'] = y_label
	checkset.to_csv('data/result/dataset_check.csv', sep=',', encoding='utf-8', index=True, index_label='Date')
	print(checkset.loc['2017-01-01':].head(12))

	# Do LDA Test
	#db.lda_reduction(dataset[dataset.columns.difference(['y'])].values, dataset['y'].values)

	# Do ADF Test
	adf = {}
	non_stationary_cols_p = []
	non_stationary_cols = []
	for col in dataset.columns.difference(['y']):
		reject, potential, res = db.adf_test(dataset.loc['2011-01-01':][col].values, 0.05, False)
		adf[col] = res
		if not reject:
			non_stationary_cols.append(col)
		if potential:
			non_stationary_cols_p.append(col)
	# Print ADF test results
	print('Non stationary columns: [%s]' % ','.join(non_stationary_cols))
	print('Potentially non stationary columns: [%s]' % ','.join(non_stationary_cols_p))
	with open('data/result/dataset_adf.json', 'w') as fp:
		json.dump(adf, fp, indent=4, sort_keys=False)