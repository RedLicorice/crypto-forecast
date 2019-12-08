import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Embedding
from keras.layers.core import Dense, Dropout
from keras.utils import to_categorical
from keras.regularizers import L1L2
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from math import ceil
import numpy as np
import matplotlib.pyplot as plt

def from_categorical(encoded):
	res = []
	for i in range(encoded.shape[0]):
		datum = encoded[i]
		decoded = np.argmax(encoded[i])
		res.append(decoded)
	return res


class Predictor:
	dataset = None
	model = None
	trainX = None
	trainY = None
	testX = None
	testY = None

	def __init__(self):
		pass

	def load_dataset(self, df, index, res, ratio = 0.7, exclude = None):
		# We want to split the dataset in train and test data
		nrows = df.shape[0]
		ntrain = ceil(nrows * ratio)
		test, train = df.tail(nrows-ntrain), df.head(ntrain)
		print("Training set size:", train.shape[0])
		print(train.head())
		print("Test set size:", test.shape[0])
		print(test.head())
		self.load_test(test, index, res, exclude)
		self.load_train(train, index, res, exclude)

	def load_test(self, df, index, res, exclude = None):
		self.testX, self.testY = self._get_xy(df, index, res, exclude)
		self.print_unique_ratio("testY", self.testY)
		self.test = df

	def load_train(self, df, index, res, exclude = None):
		self.trainX, self.trainY = self._get_xy(df, index, res, exclude)
		self.print_unique_ratio("trainY", self.trainY)
		self.train = df

	def print_unique_ratio(self, label, arr):
		total = max(1,len(arr))
		unique, counts = np.unique(arr, return_counts=True)
		for cls,cnt in zip(unique,counts):
			print("{} (class {}): count {} pct {}%".format(label, cls, cnt, cnt*100/total))

	def _get_xy(self, df, index, res, exclude = None):
		# Prepare Y
		# OneHot encode Y to get a NxM matrix where M is number of classes
		y = df[res].values
		# Prepare X
		# Drop index, result and any additional columns
		_excl = [index, res] + (exclude if exclude else [])
		features = df.drop(columns=exclude) if exclude != None else df
		# Make sure all input is numeric
		#for col in features.columns:
		#	features[col] = pd.to_numeric(features[col], errors='coerce')
		# Fill NaN values
		#features.fillna(value=0, inplace=True)
		return features.values, y # X, y

	def compile(self):
		pass

	def fit(self):
		pass

	def evaluate(self):
		pass

	def predict(self):
		if not self.model:
			raise RuntimeError("No model compiled!")
		testPredict = self.model.predict(self.testX)
		return testPredict


class SVCPredictor(Predictor):
	def compile(self, C=1.5):
		# Previously only had kernel = 'rbf
		self.model = SVC(kernel='poly', C=C, degree=30, break_ties=True)

	def fit(self):
		self.model.fit(self.trainX, self.trainY)

	def evaluate(self):
		y_pred = self.predict()
		print(confusion_matrix(self.testY, y_pred))
		print(classification_report(self.testY, y_pred))
		print("Accuracy: {}".format(accuracy_score(self.testY, y_pred)))
		pf = pd.DataFrame.from_dict({
			'predicted': y_pred,
			'expected': self.testY
		})
		pf.index = self.test.index.values
		pf.to_csv('svc_pred.csv', index_label='Date')


class SVRPredictor(Predictor):
	def compile(self):
		self.model = SVR(kernel='rbf')

	def fit(self):
		self.model.fit(self.trainX, self.trainY)

	def evaluate(self):
		y_pred = self.predict()
		print(confusion_matrix(self.testY, y_pred))
		print(classification_report(self.testY, y_pred))
		print("Accuracy: {}".format(accuracy_score(self.testY, y_pred)))
		pf = pd.DataFrame.from_dict({
			'predicted': y_pred,
			'expected': self.testY
		})
		pf.index = self.test.index.values
		pf.to_csv('svr_pred.csv', index_label='Date')


class KNNPredictor(Predictor):
	def compile(self):
		self.model = KNeighborsClassifier(n_neighbors=3)

	def fit(self):
		self.model.fit(self.trainX, self.trainY)

	def evaluate(self):
		y_pred = self.predict()
		print(confusion_matrix(self.testY, y_pred))
		print(classification_report(self.testY, y_pred))
		print("Accuracy: {}".format(accuracy_score(self.testY, y_pred)))
		pf = pd.DataFrame.from_dict({
			'predicted': y_pred,
			'expected': self.testY
		})
		pf.index = self.test.index.values
		pf.to_csv('knn_pred.csv', index_label='Date')


class DTPredictor(Predictor):
	def compile(self):
		self.model = DecisionTreeClassifier()

	def fit(self):
		self.model.fit(self.trainX, self.trainY)

	def evaluate(self):
		y_pred = self.predict()
		print(confusion_matrix(self.testY, y_pred))
		print(classification_report(self.testY, y_pred))
		print("Accuracy: {}".format(accuracy_score(self.testY, y_pred)))


class LogRegPredictor(Predictor):
	def compile(self):
		# one hot encode output
		self.trainY = to_categorical(self.trainY, num_classes=3)
		self.testY = to_categorical(self.testY, num_classes=3)

		self.model = Sequential()
		self.model.add(Dense(
						3,  # output dim is 3, one score per each class
						activation='softmax',
						kernel_regularizer=L1L2(l1=0.0, l2=0.1),
						input_dim=self.trainX.shape[1] # input dimension = number of features
						)
				  )
		self.model.compile(optimizer='sgd',
					  loss='categorical_crossentropy',
					  metrics=['accuracy'])
		#self.model.fit(self.trainX, self.trainY, epochs=100, validation_data=(self.testX, self.testY))

	def fit(self):
		if self.model is None:
			raise RuntimeError("No model compiled!")
		self.model.fit(self.trainX, self.trainY, epochs=100, batch_size=32, verbose=1, validation_data=(self.testX, self.testY))

	def evaluate(self):
		scores = self.model.evaluate(self.testX, self.testY, verbose=0)
		print("Accuracy: {}%".format(scores[1]*100))
		y_pred = self.predict()
		pf = pd.DataFrame.from_dict({
			'predicted': from_categorical(y_pred),
			'expected': from_categorical(self.testY)
		})
		pf.index = self.test.index.values
		pf.to_csv('logreg_pred.csv', index_label='Date')


class LSTMPredictor(Predictor):
	def compile(self):
		if self.trainX is None:
			raise RuntimeError("Dataset not loaded!")
		# one hot encode output
		self.trainY = to_categorical(self.trainY, num_classes=3)
		self.testY = to_categorical(self.testY, num_classes=3)
		# reshape input to be 3D [samples, timesteps, features]
		self.trainX = self.trainX.reshape((self.trainX.shape[0], 1, self.trainX.shape[1]))
		self.testX = self.testX.reshape((self.testX.shape[0], 1, self.testX.shape[1]))
		self.model = Sequential()
		#self.model.add(Embedding(self.trainX.shape[1], output_dim=256))
		self.model.add(LSTM(50, input_shape=(1, self.trainX.shape[2])))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(3)) # Should match number of categories
		self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','mse'])
		#self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

	def fit(self):
		if self.model is None:
			raise RuntimeError("No model compiled!")
		self.model.fit(self.trainX, self.trainY, epochs=100, batch_size=32, verbose=0)

	def evaluate(self):
		scores = self.model.evaluate(self.testX, self.testY, verbose=0)
		print("Accuracy: {}% MSE: {}".format(scores[1]*100, scores[2]))
		y_pred = self.predict()
		self.print_unique_ratio("predY", from_categorical(y_pred))
		pf = pd.DataFrame.from_dict({
			'predicted': from_categorical(y_pred),
			'expected': from_categorical(self.testY)
		})
		pf.index = self.test.index.values
		pf.to_csv('lstm_pred.csv', index_label='Date')


if __name__ == '__main__':
	# fix random seed for reproducibility
	np.random.seed(5)

	p = LSTMPredictor()
	df = pd.read_csv("data/result/dataset.csv", sep=',', encoding='utf-8', index_col='Date')
	input = df.loc['2011-01-01':'2013-01-01']
	p.load_dataset(input, 'Date', 'y', 0.5)
	p.compile()
	p.fit()
	# train_result = p.train()
	#test_result = p.test()
	# print(train_result)
	#print(test_result)
	p.evaluate()
	print("Done")