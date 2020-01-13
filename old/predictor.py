import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import Callback
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.model_selection import TimeSeriesSplit
import json
from matplotlib import pyplot as plt
from plotter import scatter
def from_categorical(encoded):
	res = []
	for i in range(encoded.shape[0]):
		datum = encoded[i]
		decoded = np.argmax(encoded[i])
		res.append(decoded)
	return np.array(res) + 1 # +1 because keras' from_categoric encodes from 0 while our classes start from 1

def get_unique_ratio(arr, _print=False):
	total = max(1,len(arr))
	unique, counts = np.unique(arr, return_counts=True)
	if _print:
		for cls,cnt in zip(unique,counts):
			print("{} (class {}): count {} pct {}%".format(label, cls, cnt, cnt*100/total))
	return {cls:(cnt, cnt*100/total) for cls,cnt in zip(unique,counts)}

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

	def params(self):
		return {}

	def __repr__(self):
		p = self.params()
		return self.basename+'_'+'_'.join(['{}={}'.format(k,v) for k, v in p.items()])

	def label(self):
		p = self.params()
		return self.basename + '_' + '_'.join(['{}={}'.format(k, v) for k, v in p.items()])

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

	def compile(self):
		# must return self.model
		pass

	def fit(self):
		pass

	def predict(self):
		if not self.model:
			raise RuntimeError("No model compiled!")
		testPredict = self.model.predict(self.X_test)
		return testPredict

	def evaluate(self):
		y_pred = self.predict()
		return y_pred, self.y_test


class CategoricalPredictor(Predictor):
	def load_test(self, X, y):
		Predictor.load_test(self, X, y)
		self.y_test = to_categorical(self.y_test - 1, num_classes=3)
		self.X_test = self.X_test  # self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))

	def load_train(self, X, y, **kwargs):
		Predictor.load_train(self, X, y, **kwargs)
		self.y_train = to_categorical(self.y_train - 1, num_classes=3)
		self.X_train = self.X_train  # self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))

	def predict(self):
		y_pred = Predictor.predict(self)
		return from_categorical(y_pred)

	def evaluate(self):
		y_pred = self.predict()
		return y_pred, from_categorical(self.y_test)


class SVCPredictor(Predictor):
	basename ='svc'
	kernel = 'rbf'
	degree = 3
	C = 1.0

	def __init__(self, **kwargs):
		self.kernel = kwargs.get('kernel', self.kernel)
		self.C = kwargs.get('C', self.C)
		self.degree = kwargs.get('degree', self.degree)
		Predictor.__init__(self)

	def params(self):
		return {
			'C': self.C,
			'kernel': self.kernel,
			'degree':self.degree,
		}

	def compile(self):
		# Previously only had kernel = 'rbf
		self.model = SVC(C=self.C, kernel=self.kernel, degree=self.degree)
		return self.model

	def fit(self):
		self.model.fit(self.X_train, self.y_train)


class KNNPredictor(Predictor):
	basename ='knn'
	n_neighbors = 3

	def __init__(self, **kwargs):
		self.n_neighbors = kwargs.get('n_neighbors', self.n_neighbors)
		Predictor.__init__(self)

	def params(self):
		return {
			'n_neighbors': self.n_neighbors
		}

	def compile(self):
		self.model = KNeighborsClassifier(n_neighbors=3)
		return self.model

	def fit(self):
		self.model.fit(self.X_train, self.y_train)


class MLPPredictor(Predictor):
	basename ='mlp'
	hidden_layer_sizes = (100,)
	solver = 'adam'
	learning_rate = 'constant'
	learning_rate_init = 0.001
	activation = 'relu'

	def __init__(self, **kwargs):
		self.hidden_layer_sizes = kwargs.get('hidden_layer_sizes', self.hidden_layer_sizes)
		self.solver = kwargs.get('solver', self.solver)
		self.learning_rate = kwargs.get('learning_rate', self.learning_rate)
		self.learning_rate_init = kwargs.get('learning_rate_init', self.learning_rate_init)
		self.activation = kwargs.get('activation', self.activation)
		Predictor.__init__(self)

	def params(self):
		return {
			'hidden_layer_sizes': self.hidden_layer_sizes,
			'solver': self.solver,
			'learning_rate': self.learning_rate,
			'learning_rate_init': self.learning_rate_init,
			'activation': self.activation
		}

	def compile(self):
		self.model = MLPClassifier(
			activation=self.activation,
			hidden_layer_sizes=self.hidden_layer_sizes,
			learning_rate=self.learning_rate,
			learning_rate_init=self.learning_rate_init,
			solver=self.solver
		)
		return self.model

	def fit(self):
		self.model.fit(self.X_train, self.y_train)

"""
## Commented because IDK what he wants as input format
class ExpSmoothPredictor(Predictor):
	basename ='expsmooth'
	alpha = None
	beta = None
	seasonal_periods = 5

	def __init__(self, **kwargs):
		self.alpha = kwargs.get('alpha', self.alpha)
		self.beta = kwargs.get('beta', self.beta)
		self.seasonal_periods = kwargs.get('seasonal_periods', self.beta)
		Predictor.__init__(self)

	def params(self):
		return {
			'alpha': self.alpha,
			'beta': self.beta,
			'seasonal_periods': self.seasonal_periods
		}

	def compile(self):
		r = adfuller(self.X_train.values) if self.X_train.size > 6 else adfuller(self.X_train.values, maxlag=4)
		pvalue = r[1]
		if pvalue < 0.05:
			self.model = ExponentialSmoothing(self.X_train, trend=None, seasonal=None)
		else:
			self.model = ExponentialSmoothing(self.X_train, trend='additive', seasonal='additive',
											  seasonal_periods=self.seasonal_periods)
		return self.model

	def fit(self):
		self.model.fit(smoothing_level=self.alpha, smoothing_slope=self.beta)
"""

class NNPredictor(CategoricalPredictor):
	basename ='nn'
	learning_rate = 0.001
	batch_size = 32
	epochs = 100
	metrics = ['acc', 'mse', 'mae', 'mape']

	def __init__(self, **kwargs):
		self.learning_rate = kwargs.get('learning_rate', self.learning_rate)
		self.batch_size = kwargs.get('batch_size', self.batch_size)
		self.epochs = kwargs.get('epochs', self.epochs)
		CategoricalPredictor.__init__(self)

	def params(self):
		return {
			'learning_rate':self.learning_rate,
			'batch_size':self.batch_size,
			'epochs':self.epochs
		}

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
		optimizer = Adam(learning_rate=self.learning_rate)
		self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=self.metrics)
		return self.model

	def fit(self):
		if self.model is None:
			raise RuntimeError("No model compiled!")
		self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)


class LSTMPredictor(CategoricalPredictor):
	basename ='lstm'
	timesteps = 1
	learning_rate = 0.001
	batch_size = 32
	epochs = 100
	metrics = ['acc', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error']

	def __init__(self, **kwargs):
		self.timesteps = kwargs.get('timesteps', self.timesteps)
		self.learning_rate = kwargs.get('learning_rate', self.learning_rate)
		self.batch_size = kwargs.get('batch_size', self.batch_size)
		self.epochs = kwargs.get('epochs', self.epochs)
		CategoricalPredictor.__init__(self)

	def params(self):
		return {
			'timesteps':self.timesteps,
			'learning_rate':self.learning_rate,
			'batch_size':self.batch_size,
			'epochs':self.epochs
		}

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
		self.model.add(Dense(3, activation='softmax'))
		optimizer = Adam(lr=self.learning_rate)
		self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=self.metrics)
		return self.model

	def fit(self, plot=False, to_file=None):
		if self.model is None:
			raise RuntimeError("No model compiled!")

		#In Keras the internal state is reset at the end of each batch
		# Batch size therefore represents how many states will be kept in memory.
		# Epochs, instead, determines how many times the model will be run through the training set.
		history = self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

		df = pd.DataFrame.from_dict(history.history)

		# Scale and print graph
		scaler = MinMaxScaler(feature_range=(0,1))
		df[df.columns] = scaler.fit_transform(df[df.columns])
		df['epoch'] = history.epoch
		#ax = plt.gca()
		plot = df.plot(
			kind='line',
			x='epoch',
			y=self.metrics + ['loss'],
			#ax=ax,
			figsize=(16,9)
		)
		plt.title(self.label())
		plt.savefig('plots/pred/{}.png'.format(self.label()), dpi=300)
		#plt.legend(['train', 'test'], loc='upper left')

if __name__ == '__main__':
	def get_classifiers():
		return [
			SVCPredictor(kernel='rbf'),
			#SVCPredictor(kernel='rbf', C=5.0), # Accuracy decreases
			#SVCPredictor(kernel='rbf', C=10.0), # Accuracy decreases
			#SVCPredictor(kernel='rbf', C=100.0), # Accuracy decreases
			#SVCPredictor(kernel='rbf', C=1000.0), # Accuracy decreases
			#MLPPredictor(hidden_layer_sizes=(100,), solver='adam'),
			#MLPPredictor(hidden_layer_sizes=(500,), solver='adam'),
			#MLPPredictor(hidden_layer_sizes=(1000,), solver='adam'),
			#MLPPredictor(hidden_layer_sizes=(100,), solver='sgd'),
			#MLPPredictor(hidden_layer_sizes=(500,), solver='sgd'),
			#MLPPredictor(hidden_layer_sizes=(1000,), solver='sgd'),
			#KNNPredictor(),
			#NNPredictor(),
			#LSTMPredictor(timesteps=7),
			#LSTMPredictor(timesteps=7, learning_rate=0.0005),
			#LSTMPredictor(timesteps=7,batch_size=300),
			#LSTMPredictor(timesteps=7, learning_rate=0.0005, batch_size=30, epochs=300),
			#LSTMPredictor(timesteps=14, learning_rate=0.0005, batch_size=30, epochs=300),
		]

	def expanding_window(df):
		df = df.loc['2011-01-01':]
		X = df.loc[:, df.columns != 'y'].values
		Y = df['y'].values

		n_split = 8
		classifiers = get_classifiers()
		results = {c.label():{'classifier':c,'results':{},'losses':{}} for i,c in enumerate(classifiers)}
		for i,p in enumerate(classifiers):
			step = 1
			for train_index, test_index in TimeSeriesSplit(n_splits=n_split).split(df):
				#print("Expanding window validation step: %d of %d" % (step, n_split))
				x_train, y_train = X[train_index], Y[train_index]
				x_test, y_test = X[test_index], Y[test_index]
				p.load_train(x_train, y_train)
				p.load_test(x_test, y_test)
				model = p.compile()
				p.fit()
				results[p.label()]['results'][step] = p.evaluate()
				results[p.label()]['losses'][step] = p.history.losses if hasattr(p, 'history') else []
				step += 1
		return results

	def holdout(df, _train_sz=0.7):
		classifiers = get_classifiers()
		results = {c.label(): {'classifier': c, 'results':{}, 'losses':{}} for i, c in enumerate(classifiers)}
		for i, p in enumerate(classifiers):
			test, train = train_test_split(df, train_size=_train_sz)
			#print("Training set size:", train.shape[0])
			#print(train.head())
			#print("Test set size:", test.shape[0])
			#print(test.head())

			p.load_train(train.loc[:, train.columns != 'y'].values, train['y'].values)
			p.load_test(test.loc[:, test.columns != 'y'].values, test['y'].values)

			p.compile()
			p.fit()
			results[p.label()]['results'][0] = p.evaluate()
			results[p.label()]['losses'][0] = p.history.losses if hasattr(p, 'history') else []
		return results

	# fix random seed for reproducibility
	np.random.seed(5)

	df = pd.read_csv("data/result/dataset.csv", sep=',', encoding='utf-8', index_col='Date')
	#df = add_lag(df, 14, ['y']).fillna(0)
	predictions = holdout(df.loc['2016-01-01':'2018-01-01'])

	# Save predictions report
	with open('data/result/dataset_prediction_report.txt', 'w') as fp:
		for label, info in predictions.items():
			c = info['classifier']
			r = info['results']
			l = info['losses']
			fp.write('==== Classifier: {} ====\n'.format(c.label()))
			for k,v in c.params().items():
				fp.write('{}={}\n'.format(k,v))
			fp.write('\n')
			for period, (y_pred, y) in r.items():
				fp.write('Period: {}\n'.format(period))
				if l and period in l:
					fp.write('Loss history: {}\n'.format(', '.join([str(x) for x in l[period]])))
				fp.write('Accuracy: {}%\n'.format(accuracy_score(y, y_pred)*100))
				fp.write('MSE: {}\n'.format(mean_squared_error(y, y_pred)))
				fp.write('Confusion Matrix:\n{}\n'.format(confusion_matrix(y, y_pred)))
				fp.write('Classification report:\n{}\n'.format(classification_report(y, y_pred)))
				for cls,(cnt, pct) in get_unique_ratio(y_pred).items():
					fp.write('Result class [{}] count={} pct={}%\n'.format(cls,cnt,pct))
				fp.write('\n')
			fp.write('\n')

	print("Done")