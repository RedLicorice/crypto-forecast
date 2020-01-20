import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from functools import reduce
import os
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.fftpack import fft, fftfreq

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

def correlation(corr, save_to=None, _figsz=(16,9)):
	# Set up the matplotlib figure
	f, ax = plt.subplots(figsize=_figsz)
	# Generate a custom diverging colormap
	cmap = sns.diverging_palette(220, 10, as_cmap=True)
	# Plot correlation matrix
	sns.heatmap(data=corr.round(2), annot=True, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5})
	if save_to:
		plt.savefig(save_to, dpi=72)
	plt.show()

def scatter(xcol, ycol, save_to=None, _figsz=(16,9)):
	# Set up the matplotlib figure
	f, ax = plt.subplots(figsize=_figsz)
	# Generate a custom diverging colormap
	cmap = sns.diverging_palette(220, 10, as_cmap=True)
	# Plot correlation matrix
	sns.scatterplot(x=xcol, y=ycol, cmap=cmap)
	if save_to:
		plt.savefig(save_to, dpi=300)
	plt.show()

def save_plot(name):
	os.makedirs('plots', exist_ok=True)
	plt.savefig('plots/plot_{}.png'.format(name), dpi=300)

def pca(x, y):
	pca = PCA(n_components=len(x))
	pc = pca.fit_transform(x)
	principalDf = pd.DataFrame(data=pc , columns=x.columns)
	finalDf = pd.concat([principalDf, y], axis=1)

	fig = plt.figure(figsize=(8, 8))
	ax = fig.add_subplot(1, 1, 1)
	ax.set_xlabel('Principal Component 1', fontsize=15)
	ax.set_ylabel('Principal Component 2', fontsize=15)
	ax.set_title('2 component PCA', fontsize=20)
	targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
	colors = ['r', 'g', 'b']
	for target, color in zip(targets, colors):
		indicesToKeep = finalDf['target'] == target
		ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
				   , finalDf.loc[indicesToKeep, 'principal component 2']
				   , c=color
				   , s=50)
	ax.legend(targets)
	ax.grid()

def fourier_transform(data, **kwargs):
	ylabel = kwargs.get('label', 'Input')
	if isinstance(data, pd.Series):
		ylabel = data.name
	fig, ax = plt.subplots(1, 1, figsize=(6, 3))
	data.plot(ax=ax, lw=.5)
	#ax.set_ylim(-10, 40)
	ax.set_title('{} by Date'.format(ylabel))
	ax.set_xlabel('Date')
	ax.set_ylabel(ylabel)

	ft = fft(data)
	psd = np.abs(ft) ** 2
	d = kwargs.get('d', 1/kwargs.get('sample_interval', 365)) # should be inverse of the sampling rate
	print('d='+str(d))
	ftfreq = fftfreq(len(psd), d) # 1. / 365
	i = ftfreq > 0

	fig, ax = plt.subplots(1, 1, figsize=(8, 4))
	ax.plot(ftfreq[i], 10 * np.log10(psd[i]))
	ax.set_xlim(0, 30)
	ax.set_title('{} PSD'.format(ylabel))
	ax.set_xlabel('Frequency (1/Year)')
	ax.set_ylabel('{} PSD (dB)'.format(ylabel))

	plt.show()

if __name__ == '__main__':
	data = pd.read_csv('data/atsa2017/BTC_discrete.csv', sep=',', encoding='utf-8', index_col=0)
	correlation(data.corr(), 'data/result/dataset_atsa.png')
	data = pd.read_csv('data/result/dataset.csv', sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
	correlation(data.loc['2017-01-01':'2018-01-01'].corr(), 'data/result/dataset.png')
	# data['Date'] = pd.to_datetime(data['Date'])
	#lineplot(data, ['BTC','BTC_DiffMean', 'BTC_AdrActCnt','BTC_BlkSizeByte', 'BTC_TxTfrCnt'])
	#lineplot(data, ['BTC','BTC_DiffMean', 'BTC_CapMrktCurUSD', 'BTC_CapRealUSD'])
	#lineplot(data, ['BTC','BTC_DiffMean', 'BTC_FeeMeanUSD','BTC_IssTotUSD'])
	#lineplot(data, ['BTC','BTC_IssContUSD', 'BTC_IssContNtv'])
	#lineplot(data, ['BTC','BTC_Open', 'BTC_High', 'BTC_Low'])