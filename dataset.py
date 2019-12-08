import pandas as pd
from functools import reduce
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy.signal import detrend
from technical_indicators import *
from math import isnan
from statsmodels.tsa.stattools import adfuller
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

	def add_blockchain_data(self, input, chain, symbol):
		# Rename index for join
		map = {'date': 'Date'}
		for col in chain.columns:
			if str(col) == symbol or str(col) == 'date':
				continue
			map[str(col)] = '{symbol}_{col}'.format(symbol=symbol, col=str(col))
		chain.rename(columns=map, inplace=True)
		return input.join(chain, how='left', sort=True)

	def add_ta_features(self, df, symbol):
		col_close = symbol
		col_open = symbol + '_Open'
		col_high = symbol + '_High'
		col_low = symbol + '_Low'
		col_volume = symbol + '_Volume'

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
		df[symbol+'_stoch'] = percent_k(df[col_high].values, df[col_low].values, df[col_close].values, 14)

		# Chande Momentum Oscillator
		## Not available in ta
		df[symbol+'_cmo'] = chande_momentum_oscillator(df[col_close].values, 14)

		# Average True Range Percentage
		df[symbol+'_atrp'] = average_true_range_percent(df[col_close].values, 14)

		# Percentage Volume Oscillator
		## Not available in ta
		# Settings in paper are wrong!
		df[symbol+'_pvo'] = volume_oscillator(df[col_volume].values, 12, 26)

		# Accumulation Distribution Line
		df[symbol+'_adi'] = accumulation_distribution(df[col_close].values,  df[col_high].values, df[col_low].values, df[col_volume].values)

		# On Balance Volume
		df[symbol+'_obv'] = on_balance_volume(df[col_close].values, df[col_volume].values)

		return df

	## Operations on columns
	def check_dataset(self, df):
		for col in df.columns:
			self.check_integrity(df, col)

	def check_integrity(self, df, col, fix=True):
		for i,v in enumerate(df[col].values):
			if not v:
				print("{} contains zeroes at index {} - {}".format(col, i, df.index[i]))
				if fix:
					df[col] = self.fill_holes(df[col].values)
			if v == None:
				print("{} contains None at index {} - {}".format(col, i, df.index[i]))
			if isnan(v):
				print("{} contains NaN at index {} - {}".format(col, i, df.index[i]))

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

	def drop_empty_rows(self, df, col):
		return df[df[col] != 0]

	def fill_holes(self, values, hole=np.nan):
		imp = SimpleImputer(missing_values=hole, strategy='mean')
		old_shape = values.shape
		resh = np.reshape(values, (-1,1))
		imp.fit(resh)
		res = imp.transform(resh)
		res = np.reshape(res, old_shape)
		return res

	def to_discrete(self, values, range_down = 0.01, range_up = 0.01):
		# mark a vector of variations as increase or decrease
		# if variation > range_up => increase
		# if variation < range_down => decrease
		# else if range_down < variation < range_up => stationary
		y = []
		for v in values:
			if v > range_up:  # mark as Increase
				y.append(2)
			elif v < range_down:  # mark
				y.append(0)
			else:  # by default we hodl (1)
				y.append(1)
		return y

	def discrete_label(self, values):
		y = []
		for v in values:
			if v == 0:
				y.append('Decrease')
			elif v == 1:
				y.append('Stationary')
			elif v == 2:
				y.append('Increase')
		return y


	def to_variation(self, input, shift = -1):
		# if shift > 0, np.roll gives t - shift
		# if shift < 0, np.roll gives t + shift
		## Example
		# x = 				0 1 2 3 4 5 6 7 8 9
		# np.roll(x, 1) = 	9 0 1 2 3 4 5 6 7 8
		# np.roll(x, -1) =	1 2 3 4 5 6 7 8 9 0

		shifted = np.roll(input, shift)
		shifted = self.fill_holes(shifted)
		#np.seterr(divide='ignore')
		if shift > 0:
			# variation with past value
			# input is t+1
			# shift is t
			diff = input - shifted
			return np.divide(diff, shifted) # Pt+1 - Pt / Pt
		if shift < 0:
			# variation with future value
			# input is t
			# shift is t+1
			diff = shifted - input
			return np.divide(diff, input)  # Pt+1 - Pt / Pt

	def to_difference(self, input, shift = 1):
		shifted = np.roll(input, shift)
		shifted = self.fill_holes(shifted)

		diff = input - shifted
		if shift < 0:
			diff = shifted - input
		return diff

	def add_window(self, df, column, length):
		if not column in df.columns:
			raise ('Column does not exist!')
		for i in range(length):
			df[column+'-'+str(i+1)] = np.roll(df[column].values, (i+1))
		return df

	def _shape(self, values, range=(0,1)):
		scaler = MinMaxScaler(feature_range=range)
		# df["Close"] = pd.to_numeric(df["Close"])
		shaped = np.reshape(values, (-1, 1))
		scaled = scaler.fit_transform(shaped)
		return scaled

	def scale(self, df, exclude = None, range=(0,1)):
		for col in df.columns.difference(exclude) if exclude != None else df.columns:
			#numeric = pd.to_numeric(df[col], errors='coerce')
			df[col] = self._shape(df[col].values, range)
		return df

	## Feature selection
	"""""
	def select_features_genetic(self, df, exclude):
		#df['y_var'] = pd.to_numeric(df['y_var'])
		#df.fillna(0, inplace=True)
		y = df['y'].values
		feature_count = len(df.columns) - len(exclude)
		features = df.drop(columns=exclude)
		for col in features.columns:
			numeric = pd.to_numeric(features[col], errors='coerce')
			features[col] = self._shape(numeric.values)
		features.fillna(value=0, inplace=True)
		X = features.values
		estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")
		selector = GeneticSelectionCV(estimator,
					cv=5,
					verbose=1,
					scoring="accuracy",
					max_features=20,
					n_population=150,
					crossover_proba=0.5,
					mutation_proba=0.2,
					n_generations=300,
					crossover_independent_proba=0.5,
					mutation_independent_proba=0.05,
					tournament_size=3,
					n_gen_no_change=10,
					caching=True,
					n_jobs=-1
		)
		selector = selector.fit(X, y)

		selected_features = selector.support_
		final_features = []
		for idx,col in enumerate(features.columns):
			if col.startswith('Unnamed'):
				print("Unnnamed column:{}".format(idx))
			if selected_features[idx]:
				final_features.append(col)


		print(final_features)
		return final_features
		"""""

# Testing, this is NOT a driver
if __name__ == '__main__':
	#np.seterr(all='raise')
	db = DatasetBuilder()
	# Join OHLC and Volume datasets for each year
	datasets = []
	for year in [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]:
		print('Year: '+str(year))
		ohlc = pd.read_csv('data/polito/'+str(year)+'_candle.csv', sep=',', index_col='Date')
		vol = pd.read_csv('data/polito/'+str(year)+'_volume.csv', sep=',', index_col='Date')
		ohlcv = db.make_ohlcv('BTC', ohlc, vol)
		datasets.append(ohlcv)
	# Merge yearly datasets
	ohlcv = reduce(lambda left, right: left.append(right), datasets)
	print("OHLCV rows: " + str(ohlcv.shape[0]))

	# Build the dataset
	dataset = db.add_ta_features(ohlcv, 'BTC')
	print("OHLCV+TA rows: " + str(dataset.shape[0]))

	# Add blockchain information
	#chain = pd.read_csv('data/coinmetrics.io/btc.csv', sep=',', index_col='date')
	#chain.drop(columns=['PriceUSD'], inplace=True) # Drop PriceUSD because it's redundant
	#chain.dropna(axis=1, how='all', inplace=True) # Drop columns composed of NaN values
	#dataset = db.add_blockchain_data(dataset, chain, 'BTC')
	#print("OHLCV+TA+Blockchain rows: " + str(dataset.shape[0]))

	# Add Y
	values = db.fill_holes(dataset['BTC'].values)
	y = db.to_variation(values, -1)  # variation w.r.t. next day
	y = db.to_discrete(y, -0.01, 0.01)
	dataset['y'] = y

	# Fill holes by average, from relevant records only (previous ones may contain holes due to windows spinning in)
	db.check_dataset(dataset[dataset.columns.difference(['y'])].loc['2011-01-01':])

	# Make data stationary by differencing
	for col in ['BTC','BTC_High','BTC_Low','BTC_Open','BTC_adi','BTC_obv']:
		#dataset[col] = db.to_variation(dataset[col].values, 1)
		dataset[col] = db.to_difference(dataset[col].values, 1) # best results on LSTM
		#dataset[col] = detrend(dataset[col].values, axis=-1, type='linear')
	# Normalize data
	dataset = db.scale(dataset, ['y']) # Rescale everything but y in (0,1)

	# Save dataset and correlation matrix
	dataset.to_csv('data/result/dataset.csv', sep=',', encoding='utf-8', index=True, index_label='Date')
	dataset.corr().to_csv('data/result/dataset_corr.csv', sep=',', encoding='utf-8', index=True, index_label='index')

	# Print some rows to check everything is alright
	pd.set_option('display.float_format', lambda x: '%.8f' % x)
	dataset['y_var'] = db.to_variation(dataset['BTC'].values)
	dataset['y_label'] = db.discrete_label(dataset['y'].values)
	checkset = dataset[['BTC', 'y_var', 'y', 'y_label']].loc['2011-01-01':]
	checkset.to_csv('data/result/dataset_check.csv', sep=',', encoding='utf-8', index=True, index_label='Date')
	print(checkset.head(12))

	# Do ADF Test
	adf = {}
	non_stationary_cols_p = []
	non_stationary_cols = []
	for col in dataset.columns.difference(['y','y_label','y_var']):
		reject, potential, res = db.adf_test(dataset.loc['2011-01-01':][col].values, 0.05, False)
		adf[col] = res
		if not reject:
			non_stationary_cols.append(col)
		if potential:
			non_stationary_cols_p.append(col)
	print('Non stationary columns: [%s]' % ','.join(non_stationary_cols))
	print('Potentially non stationary columns: [%s]' % ','.join(non_stationary_cols_p))
	with open('data/result/dataset_adf.json', 'w') as fp:
		json.dump(adf, fp, indent=4, sort_keys=False)