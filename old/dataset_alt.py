import pandas as pd
import numpy as np
from dataset import DatasetBuilder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import seaborn as sns
from matplotlib import pyplot as plt
import json

# predict price variations on an altcoin based on btc, and some other altcoins
#
if __name__ == '__main__':
	db = DatasetBuilder()
	target = 'ETH'
	symbols = {
		'ADA': "ADA",
		'BCH': "BCH",
		'BNB': "BNB",
		'BTC': "BTC",
		'BTG': "BTG",
		'DASH': "DASH",
		'DOGE': "DOGE",
		'EOS': "EOS",
		'ETC': "ETC",
		'ETH': "ETH",
		'IOT': "MIOTA",
		'LINK': "LINK",
		'LTC': "LTC",
		'NEO': "NEO",
		'QTUM': "QTUM",
		'TRX': "TRX",
		'USDT': "USDT",
		'VEN': "VET",
		'WAVES': "WAVES",
		'XEM': "XEM",
		'XMR':"XMR",
		'XRP': "XRP",
		'ZEC': "ZEC",
		'ZRX': "ZRX"
	}
	years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
	datasets = []
	#_symbols = []
	for year in years:
		print('Year: ' + str(year))
		ohlc = pd.read_csv('data/polito/' + str(year) + '_candle.csv', sep=',', index_col='Date', parse_dates=True)
		vol = pd.read_csv('data/polito/' + str(year) + '_volume.csv', sep=',', index_col='Date', parse_dates=True)
		# Some features (2013) contain an empty column, drop it!
		ohlc = ohlc.drop(columns=db.get_non_numeric(ohlc))
		vol = vol.drop(columns=db.get_non_numeric(vol))

		merge_col = [x for x in vol.columns if x.endswith('_Volume')]
		print("Merging:",merge_col)
		ohlcv = ohlc.merge(vol[merge_col], how='outer', sort=True, left_index=True, right_index=True)
		#_symbols = _symbols + [x.split('_')[0] for x in ohlcv.columns if x.endswith('_Open')]
		datasets.append(ohlcv)
	dataset = pd.concat(datasets, sort=True)

	db.change_ohlcv_symbol(dataset, mapping=symbols)
	print("OHLCV rows: " + str(dataset.shape[0]))
	dataset = dataset.fillna(value=0)

	#print(set(_symbols))
	for symbol in symbols.values():
		# Build the dataset
		dataset = db.add_ta_features(dataset, symbol)

		# Add blockchain information
		if symbol == 'MIOTA': # No blockchain data for miota
			continue
		chain = pd.read_csv('data/coinmetrics.io/{}.csv'.format(symbol), sep=',', index_col='date', parse_dates=True)
		chain.drop(columns=['PriceUSD'], inplace=True)  # Drop PriceUSD because it's redundant
		chain.dropna(axis=0, how='all', inplace=True)  # Drop columns composed of NaN values
		chain.dropna(axis=1, how='all', inplace=True)  # Drop rows composed of NaN values
		clean_chain = chain.replace([np.inf, -np.inf], np.nan).interpolate(method='linear', axis=1)
		dataset = db.add_blockchain_data(dataset, clean_chain, symbol)

	# Add Y
	y_ref = dataset[[target]].copy().values
	y_var = dataset[target].pct_change(1).fillna(0)  # variation w.r.t. next day
	y = db.to_discrete(y_var, -0.01, 0.01)
	y_label = db.discrete_label(y)

	# Fill holes linearly
	dataset = dataset.replace([np.inf, -np.inf], np.nan).interpolate(method='linear', axis=1)

	# Normalize data
	scaler = RobustScaler()
	scaled = scaler.fit_transform(dataset.values)
	dataset = pd.DataFrame(scaled, columns=dataset.columns, index=dataset.index)

	# Append target
	dataset['y'] = y

	# Save dataset and correlation matrix
	dataset.to_csv('data/result/dataset_alt.csv', sep=',', encoding='utf-8', index=True, index_label='Date')
	corr = dataset.corr()
	corr.to_csv('data/result/dataset_alt_corr.csv', sep=',', encoding='utf-8', index=True, index_label='index')

	# Print some rows to check everything is alright
	pd.set_option('display.float_format', lambda x: '%.12f' % x)
	checkset = pd.DataFrame(index=dataset.index)
	checkset['close'] = y_ref
	checkset['y'] = y
	checkset['y_var'] = y_var
	checkset['y_label'] = y_label
	checkset.to_csv('data/result/dataset_check.csv', sep=',', encoding='utf-8', index=True, index_label='Date')
	print(checkset.loc['2017-01-01':].head(12))

	# Do LDA Test
	# db.lda_reduction(dataset[dataset.columns.difference(['y'])].values, dataset['y'].values)

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