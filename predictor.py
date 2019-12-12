import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.regularizers import L1L2
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from math import ceil
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.model_selection import TimeSeriesSplit

def from_categorical(encoded):
	res = []
	for i in range(encoded.shape[0]):
		datum = encoded[i]
		decoded = np.argmax(encoded[i])
		res.append(decoded)
	return np.array(res)

def add_lag(df, lag, exclude):
	lags = range(1,lag)
	return df.assign(**{
		'{}_-{}'.format(col, t): df[col].shift(t)
		for t in lags
		for col in df
		if col not in exclude
	})

def get_bench_dataset():
	# Test algos on a well-known dataset
	# https://janakiev.com/notebooks/keras-iris/
	from sklearn.preprocessing import StandardScaler
	iris = load_iris()
	df = pd.DataFrame(data=iris['data'], columns=['sepal_length','sepal_width','petal_length','petal_width'])
	y = iris['target']
	scaler = StandardScaler()
	scaled = scaler.fit_transform(df.values)
	input = pd.DataFrame(scaled, columns=df.columns, index=df.index)
	input['y'] = y
	return input.sample(frac=1) # Shuffle the dataset

class Predictor:
	basename ='predictor'
	dataset = None
	model = None
	X_train = None
	y_train = None
	X_test = None
	y_test = None
	testIdx = None

	def __init__(self):
		pass

	def load_test(self, X, y):
		self.X_test = X
		self.y_test = y

	def load_train(self, X, y, **kwargs):
		if kwargs.get('oversample',False):
			X, y = self.oversample(X, y)
		elif kwargs.get('undersample', False):
			X, y = self.undersample(X, y)
		self.X_train = X
		self.y_train = y

	def oversample(self, X, y):
		sm = SMOTE(random_state=12)
		return sm.fit_sample(X, y)

	def undersample(self, X, y):
		cc = ClusterCentroids(random_state=12)
		return cc.fit_resample(X, y)

	def print_input_stats(self):
		self.print_unique_ratio("y_train", self.y_train)
		self.print_unique_ratio("y_test", self.y_test)

	def print_unique_ratio(self, label, arr):
		if hasattr(arr, 'shape') and len(arr.shape) > 1 and arr.shape[1]:
			arr = from_categorical(arr)
		total = max(1,len(arr))
		unique, counts = np.unique(arr, return_counts=True)
		for cls,cnt in zip(unique,counts):
			print("{} (class {}): count {} pct {}%".format(label, cls, cnt, cnt*100/total))

	def compile(self):
		# must return self.model
		pass

	def fit(self):
		pass

	def evaluate(self):
		pass

	def evaluate_report(self, y_pred, y_test, index = None):
		pf = pd.DataFrame.from_dict({
			'predicted': np.reshape(y_pred, (-1,)),
			'expected': y_test
		})
		pf.index = pd.RangeIndex() if index is None else index
		pf.to_csv('{}_pred.csv'.format(self.basename), index_label='Date')

	def predict(self):
		if not self.model:
			raise RuntimeError("No model compiled!")
		testPredict = self.model.predict(self.X_test)
		return testPredict


class SVCPredictor(Predictor):
	basename ='svc'

	def compile(self):
		# Previously only had kernel = 'rbf
		self.model = SVC(kernel='rbf')
		return self.model

	def fit(self):
		self.model.fit(self.X_train, self.y_train)

	def evaluate(self, index=None):
		y_pred = self.predict()
		print(confusion_matrix(self.y_test, y_pred))
		print(classification_report(self.y_test, y_pred))
		print("Accuracy: {}".format(accuracy_score(self.y_test, y_pred)))
		self.evaluate_report(y_pred, self.y_test, index)
		return accuracy_score(self.y_test, y_pred)


class KNNPredictor(Predictor):
	basename ='knn'

	def compile(self):
		self.model = KNeighborsClassifier(n_neighbors=3)
		return self.model

	def fit(self):
		self.model.fit(self.X_train, self.y_train)

	def evaluate(self, index=None):
		y_pred = self.predict()
		print(confusion_matrix(self.y_test, y_pred))
		print(classification_report(self.y_test, y_pred))
		print("Accuracy: {}".format(accuracy_score(self.y_test, y_pred)))
		self.evaluate_report(y_pred, self.y_test, index)
		return accuracy_score(self.y_test, y_pred)


class NNPredictor(Predictor):
	basename ='nn'

	def load_test(self, X, y):
		Predictor.load_test(self, X, y)
		self.y_test = to_categorical(self.y_test - 1, num_classes=3)
		self.X_test = self.X_test # self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))

	def load_train(self, X, y, **kwargs):
		Predictor.load_train(self, X, y, **kwargs)
		self.y_train = to_categorical(self.y_train - 1, num_classes=3)
		self.X_train = self.X_train # self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))

	def compile(self):
		# one hot encode output
		self.model = Sequential()
		self.model.add(Dense(
						32,
						input_dim=self.X_train.shape[1],
						activation='relu',
						#kernel_regularizer=L1L2(l1=0.5, l2=0.1)
						)
				  )

		self.model.add(Dense(
						3,  # output dim is 3, one score per each class
						activation='softmax',
						#kernel_regularizer=L1L2(l1=0.2, l2=0.4),
						input_dim=self.X_train.shape[1] # input dimension = number of features
						)
				  )

		self.model.compile(optimizer='adam',
					  loss='binary_crossentropy',
					  metrics=['accuracy'])
		return self.model

	def fit(self):
		if self.model is None:
			raise RuntimeError("No model compiled!")
		self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=64, verbose=0, validation_data=(self.X_test, self.y_test))

	def evaluate(self, index = None):
		scores = self.model.evaluate(self.X_test, self.y_test, verbose=0)
		print("Accuracy: {}%".format(scores[1]*100))
		y_pred = self.predict()
		self.evaluate_report(from_categorical(y_pred), from_categorical(self.y_test), index)
		return scores[1]


class LSTMPredictor(Predictor):
	basename ='lstm'
	timesteps = 1

	def __init__(self, timesteps = 1):
		self.timesteps = timesteps
		Predictor.__init__(self)

	def split_sequences(self, X, y, n_steps):
		_X, _y = [], []
		for i in range(len(X)):
			end_ix = i + n_steps
			if(end_ix > len(X)):
				break
			seq_x = X[i:end_ix, :]
			seq_y = y[end_ix-1]
			_X.append(seq_x)
			_y.append(seq_y)
		return np.array(_X), np.array(_y)

	def load_test(self, X, y):
		Predictor.load_test(self, X, y)
		_X, _y = self.split_sequences(self.X_test, self.y_test, self.timesteps)
		self.y_test = to_categorical(_y - 1, num_classes=3)
		self.X_test = _X #self.X_test.reshape((self.X_test.shape[0], self.timesteps, self.X_test.shape[1]))

	def load_train(self, X, y, **kwargs):
		Predictor.load_train(self, X, y, **kwargs)
		_X, _y = self.split_sequences(self.X_train, self.y_train, self.timesteps)
		self.y_train = to_categorical(_y - 1, num_classes=3)
		self.X_train = _X # self.X_train.reshape((self.X_train.shape[0], self.timesteps, self.X_train.shape[1]))

	def compile(self):
		if self.X_train is None:
			raise RuntimeError("Dataset not loaded!")

		self.model = Sequential()
		self.model.add(LSTM(50, input_shape=(self.timesteps, self.X_train.shape[2]), activation='tanh'))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(3, activation='sigmoid')) # Should match number of categories
		optimizer = Adam(learning_rate=0.0001)
		self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy','mse'])
		return self.model

	def fit(self):
		if self.model is None:
			raise RuntimeError("No model compiled!")
		#In Keras the internal state is reset at the end of each batch
		# Batch size therefore represents how many states will be kept in memory.
		# Epochs, instead, determines how many times the model will be run through the training set.
		self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=32, verbose=1)

	def evaluate(self, index = None):
		scores = self.model.evaluate(self.X_test, self.y_test, verbose=0)
		print("Accuracy: {}% MSE: {}".format(scores[1]*100, scores[2]))
		y_pred = self.predict()

		self.evaluate_report(from_categorical(y_pred), from_categorical(self.y_test), index[self.timesteps-1:])
		return scores[1]


if __name__ == '__main__':

	def expanding_window(df):
		p = LSTMPredictor(timesteps=7)
		df = df.loc['2011-01-01':]
		X = df.loc[:, df.columns != 'y'].values
		Y = df['y'].values

		n_split = 8
		step = 1
		for train_index, test_index in TimeSeriesSplit(n_splits=n_split).split(df):
			print("Expanding window validation step: %d of %d" % (step, n_split))
			x_train, y_train = X[train_index], Y[train_index]
			x_test, y_test = X[test_index], Y[test_index]
			p.load_train(x_train, y_train, oversample=True)
			p.load_test(x_test, y_test)
			model = p.compile()
			p.fit()
			p.evaluate(test_index)
			step += 1

	def holdout(df):
		p = LSTMPredictor()
		test, train = train_test_split(df.loc['2016-01-01':'2018-01-01'], train_size=0.6)
		print("Training set size:", train.shape[0])
		print(train.head())
		print("Test set size:", test.shape[0])
		print(test.head())

		p.load_train(train.loc[:, train.columns != 'y'].values, train['y'].values, oversample=True)
		p.load_test(test.loc[:, test.columns != 'y'].values, test['y'].values)
		p.print_input_stats()
		p.compile()
		p.fit()
		p.evaluate(test.index)

		validation = df.loc['2011-01-01':'2015-01-01']

		p.load_test(validation.loc[:, validation.columns != 'y'].values, validation['y'].values)
		p.evaluate(validation.index)


	# fix random seed for reproducibility
	np.random.seed(5)


	df = pd.read_csv("data/result/dataset.csv", sep=',', encoding='utf-8', index_col='Date')
	# df = add_lag(df, 14, ['y'])
	expanding_window(df)

	print("Done")