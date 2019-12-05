import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Dropout

class Predictor:
	dataset = None
	model = None
	trainX = None
	trainY = None
	testX = None
	testY = None

	def __init__(self):
		pass

	def load_dataset(self, df, index, result, exclude = None):
		# We want to split the dataset in train and test data (50%) - returns dataframes since 0.16
		train, test = train_test_split(df, test_size=0.5)
		self.trainX, self.trainY = self._get_xy(train, index, result, exclude)
		self.testX, self.testY = self._get_xy(train, index, result, exclude)

	def _get_xy(self, df, index, result, exclude = None):
		y = df[result].values
		_excl = [index, result] + (exclude if exclude else [])
		features = df.drop(columns=exclude)
		for col in features.columns:
			features[col] = pd.to_numeric(features[col], errors='coerce')
		features.fillna(value=0, inplace=True)
		X = features.values
		return X, y

	def compile_lstm_model(self):
		if self.trainX is None:
			raise RuntimeError("Dataset not loaded!")
		self.model = Sequential()
		self.model.add(LSTM(50, input_shape=self.trainX.shape))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(1))
		self.model.compile(loss='mse', optimizer='adam')

	def fit(self):
		if self.model is None:
			raise RuntimeError("No model compiled!")
		self.model.fit(self.trainX, self.trainY, epochs=100, batch_size=24, verbose=1)

	def train(self):
		if not self.model:
			raise RuntimeError("No model compiled!")
		trainPredict = self.model.predict(self.trainX)
		return trainPredict

	def test(self):
		if not self.model:
			raise RuntimeError("No model compiled!")
		testPredict = self.model.predict(self.testX)
		return testPredict

if __name__ == '__main__':
	p = Predictor()
	df = pd.read_csv("data/result/btc_ohlcv_reduced.csv", sep=',', encoding='utf-8', index_col='Date')
	p.load_dataset(df, 'Date', 'y', ['y_var'])
	p.compile_lstm_model()
	p.fit()
	p.train()
	p.test()