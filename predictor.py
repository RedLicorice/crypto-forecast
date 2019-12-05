import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout
from keras.utils import to_categorical
from math import ceil

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
		nrows = df.shape[0]
		ntest = ceil(nrows / 2)
		test, train = df.head(ntest), df.tail(ntest)
		print("Training set")
		print(train.head())
		print("Test set")
		print(test.head())
		trainX, self.trainY = self._get_xy(train, index, result, exclude)
		testX, self.testY = self._get_xy(train, index, result, exclude)
		# reshape input to be 3D [samples, timesteps, features]
		self.trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
		self.testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))

	def _get_xy(self, df, index, result, exclude = None):
		# Prepare Y
		y = to_categorical(df[result].values, num_classes=3)
		# Prepare X
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
		self.model.add(LSTM(50, input_shape=(1, self.trainX.shape[2])))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(3)) # Should match number of categories
		self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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

	def evaluate(self):
		scores = self.model.evaluate(self.testX, self.testY, verbose=0)
		print("Accuracy: %.2f%%" % (scores[1]*100))

if __name__ == '__main__':
	p = Predictor()
	df = pd.read_csv("data/result/btc_all_scaled.csv", sep=',', encoding='utf-8', index_col='Date')
	p.load_dataset(df, 'Date', 'y', ['y_var'])
	p.compile_lstm_model()
	p.fit()
	train_result = p.train()
	#test_result = p.test()
	print(train_result)
	#print(test_result)
	p.evaluate()
	print("Done")