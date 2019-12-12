import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from functools import reduce
import os
import seaborn as sns
from matplotlib import pyplot as plt

def minmax_scaler(values, range=(0,1)):
	scaler = MinMaxScaler(feature_range=range)
	shaped = np.reshape(values, (-1, 1))
	scaled = scaler.fit_transform(shaped)
	return scaled

def standard_scaler(values):
	scaler = StandardScaler()
	shaped = np.reshape(values, (-1, 1))
	scaled = scaler.fit_transform(shaped)
	return scaled

def lineplot(df, y, _scale = False):
	if _scale:
		for col in df.columns.difference(['Date']):
			df[col] = standard_scaler(df[col].values)
	ax = plt.gca()
	plot = df.plot(
		kind='line',
		x='Date',
		y=y,
		ax=ax,
		figsize=(24,20)
	)
	save_plot('_'.join(y))
	plt.show()

def correlation(corr, save_to=None):
	# Set up the matplotlib figure
	f, ax = plt.subplots(figsize=(40, 42))
	# Generate a custom diverging colormap
	cmap = sns.diverging_palette(220, 10, as_cmap=True)
	# Plot correlation matrix
	sns.heatmap(data=corr.round(2), annot=True, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5})
	if save_to:
		plt.savefig(save_to, dpi=300)
	plt.show()

def save_plot(name):
	os.makedirs('plots', exist_ok=True)
	plt.savefig('plots/plot_{}.png'.format(name), dpi=300)

if __name__ == '__main__':
	data = pd.read_csv('data/result/dataset_alt_corr.csv', sep=',', encoding='utf-8', index_col='index')
	correlation(data, 'data/result/dataset_alt_corr.png')
	# data['Date'] = pd.to_datetime(data['Date'])
	#lineplot(data, ['BTC','BTC_DiffMean', 'BTC_AdrActCnt','BTC_BlkSizeByte', 'BTC_TxTfrCnt'])
	#lineplot(data, ['BTC','BTC_DiffMean', 'BTC_CapMrktCurUSD', 'BTC_CapRealUSD'])
	#lineplot(data, ['BTC','BTC_DiffMean', 'BTC_FeeMeanUSD','BTC_IssTotUSD'])
	#lineplot(data, ['BTC','BTC_IssContUSD', 'BTC_IssContNtv'])
	#lineplot(data, ['BTC','BTC_Open', 'BTC_High', 'BTC_Low'])