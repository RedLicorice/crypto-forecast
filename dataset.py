import pandas as pd
import ta
from functools import reduce
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from genetic_selection import GeneticSelectionCV


class DatasetBuilder:
	def __init__(self):
		pass

	def make_ohlcv(self, symbol, candle, volume):
		ohlc = pd.read_csv(candle, sep=',')
		vol = pd.read_csv(volume, sep=',')
		ohlcv = pd.merge(ohlc, vol[['Date', symbol+'_Volume']], on='Date', how='left', sort=True)
		# Drop all columns except the ones related to this symbol
		keep = ['Date']
		for col in ohlcv.columns:
			if symbol in col:
				keep.append(col)
		ohlcv.drop(ohlcv.columns.difference(keep), 1, inplace=True)
		return ohlcv

	def add_blockchain_data(self, input, symbol, chaindata, metrics = None):
		chain = pd.read_csv(chaindata, sep=',')
		# Rename index for join
		map = {'date': 'Date'}
		for col in chain.columns:
			if str(col) == symbol or str(col) == 'date':
				continue
			map[str(col)] = '{symbol}_{col}'.format(symbol=symbol, col=str(col))
		chain.rename(columns=map, inplace=True)
		return pd.merge(input, chain[chain.columns], on='Date', how='left', sort=True)

	def add_ta_features(self, input, symbol):
		df = ta.add_all_ta_features(input, symbol + "_Open",  symbol + "_High",  symbol + "_Low",  symbol,  symbol + "_Volume", fillna=True)
		map = {}
		for col in df.columns:
			if not symbol in col and not 'Date' in col:
				map[col] = symbol + "_" + col
		df.rename(columns=map, inplace=True)
		return df

	def add_y(self, input, symbol, shift, threshold_down, threshold_up):
		close = input[symbol].values
		forecast = np.roll(close, -shift)
		# get price difference
		diff = forecast - close
		# divide to get variation (percent in 0-1)
		variation = np.divide(diff, close)
		#variation.fillna(inplace = True, value=0)
		y = []
		for v in variation:
			if v > threshold_up: # if price will rise tomorrow, we buy (1)
				y.append(1)
			elif v < threshold_down: # if price will drop tomorrow, we sell (2)
				y.append(2)
			else: # by default we hodl (0)
				y.append(0)
		#shaped = np.reshape(values, (-1, 1))
		#scaled = scaler.fit_transform(shaped)
		input['y'] = y
		input['y_var'] = variation

		return input

	def _shape(self, values, range=(0,1)):
		scaler = MinMaxScaler(feature_range=range)
		# df["Close"] = pd.to_numeric(df["Close"])
		shaped = np.reshape(values, (-1, 1))
		scaled = scaler.fit_transform(shaped)
		return scaled

	def scale(self, df, exclude = None, range=(0,1)):
		for col in df.columns.difference(exclude) if exclude else df.columns:
			numeric = pd.to_numeric(df[col], errors='coerce')
			df[col] = self._shape(numeric.values, range)
		return df

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


# Testing, this is NOT a driver
if __name__ == '__main__':
	db = DatasetBuilder()
	# Join OHLC and Volume datasets for each year
	datasets = []
	for year in [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]:
		print('Year: '+str(year))
		ohlcv = db.make_ohlcv('BTC', 'data/polito/'+str(year)+'_candle.csv', 'data/polito/'+str(year)+'_volume.csv')
		datasets.append(ohlcv)
	# Merge yearly datasets
	main = reduce(lambda left, right: left.append(right), datasets)
	main.fillna(0, inplace=True)
	# Enrich merged dataset with TA and Blockchain features
	main = db.add_blockchain_data(main, 'BTC', 'data/coinmetrics.io/btc.csv')
	main = db.add_ta_features(main, 'BTC')
	# Determine target classes
	db.add_y(main, 'BTC', 1, 0.01, 0.01) # use BTC price one day ahead, buy if
	main.to_csv('data/result/btc_all.csv', sep=',', encoding='utf-8', index=False)
	# Scale dataset (all values must be in the same range for genetic reduction)
	scaled = db.scale(main, exclude=['Date','y','y_var'])
	scaled.to_csv('data/result/btc_all_scaled.csv', sep=',', encoding='utf-8', index=False)
	# Select features using genetic search
	sel_features = db.select_features_genetic(scaled, ['Date','y','y_var'])
	# Save scaled dataset
	reduced = scaled.drop(scaled.columns.difference(['Date','y','y_var']+sel_features), 1)
	reduced.to_csv('data/result/btc_ohlcv_reduced.csv', sep=',', encoding='utf-8', index=False)
