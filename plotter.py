import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from functools import reduce
import os

class Plotter:
	ax = None

	def __init__(self):
		self.ax = plt.gca()

	def scale(self, values, range=(0,1)):
		scaler = MinMaxScaler(feature_range=range)
		shaped = np.reshape(values, (-1, 1))
		scaled = scaler.fit_transform(shaped)
		return scaled

	def lineplot(self, df, y, scale = False):
		if scale:
			for col in df.columns.difference(['Date']):
				df[col] = self.scale(df[col].values)
		ax = plt.gca()
		plot = df.plot(
			kind='line',
			x='Date',
			y=y,
			ax=ax,
			figsize=(24,20)
		)
		self.save_plot('_'.join(y))
		plt.show()

	def correlation(self, df):
		corr = df.corr()
		print(corr)
		plt.matshow(corr)
		self.save_plot('correlation')

	def save_plot(self, name):
		os.makedirs('plots', exist_ok=True)
		plt.savefig('plots/plot_{}.png'.format(name), dpi=300)

if __name__ == '__main__':
	p = Plotter()
	data = pd.read_csv('data/result/btc_rolled.csv', sep=',', encoding='utf-8', index_col=None)
	data['Date'] = pd.to_datetime(data['Date'])
	#p.lineplot(data, ['BTC','BTC_DiffMean', 'BTC_AdrActCnt','BTC_BlkSizeByte', 'BTC_TxTfrCnt'])
	#p.lineplot(data, ['BTC','BTC_DiffMean', 'BTC_CapMrktCurUSD', 'BTC_CapRealUSD'])
	#p.lineplot(data, ['BTC','BTC_DiffMean', 'BTC_FeeMeanUSD','BTC_IssTotUSD'])
	#p.lineplot(data, ['BTC','BTC_IssContUSD', 'BTC_IssContNtv'])
	p.lineplot(data, ['BTC','BTC_Open', 'BTC_High', 'BTC_Low'])