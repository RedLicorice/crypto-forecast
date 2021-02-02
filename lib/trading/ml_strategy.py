#
#   This class handles Signal generation for a given symbol
#   A strategy instance is needed for each symbol.
#   Normal usage would be:
#       strat = Strategy('pipeline', 'dataset', 'symbol')
#       for day in days:
#           signal = strat.get_signal(day)
#
import importlib
import numpy as np
import os
from lib.dataset import load_symbol
from lib.trading.models import Signal, SignalType
import json, pickle

class MLStrategy:
    cache_dir = None
    pipeline = None # The pipeline module we're using for predictions
    parameters = None # Parameters (dict) for training the pipeline
    symbol = None # Symbol we're generating signals for
    _pipeline = None # pipeline name, used for repr
    _dataset = None # dataset name, used for repr
    _target = None # target type, used for repr

    def __init__(self, pipeline, dataset, symbol, target='class', parameters=None, window_size=150, skip_cache=False, cache_base_dir='./equities'):
        # Import the pipeline module
        self.pipeline = importlib.import_module('pipelines.' + pipeline)
        # Load the whole dataset for this symbol
        self._features, self._target, self._close = load_symbol(dataset, symbol, target=target)
        # Drop nan rows from features
        self._features = self._features.dropna(axis='columns', how='all').replace([np.inf, -np.inf], np.nan)
        # Determine model cache path, and create it
        if not skip_cache:
            self.cache_dir = '{}/models/{}_{}_{}_w{}/{}/'.format(cache_base_dir, pipeline, dataset, target, window_size, symbol)
            os.makedirs(self.cache_dir, exist_ok=True)
        # If parameters is a path to json file, open and deserialize it
        if parameters.endswith('.json') and os.path.exists(parameters):
            with open(parameters, 'r') as f:
                self.parameters = json.load(f)
        # If parameters is a path to pickle file, open and load it
        elif parameters.endswith('.p') and os.path.exists(parameters):
            with open(parameters, 'rb') as f:
                self.parameters = pickle.load(f)
        # If parameters is a dict, assign it
        elif isinstance(parameters, dict):
            self.parameters = parameters
        # Save instance parameters
        self.symbol = symbol
        self.window_size = window_size

    def __repr__(self):
        return 'MLStrategy(symbol={}, dataset={}, target={}, pipeline={})'.format(self.symbol, self._dataset, self._target, self._pipeline)

    def get_data(self, day, train_window_size):
        if day not in self._features.index:
            raise Exception('No data on symbol {} for day {}!'.format(self.symbol, day))
        i = self._features.index.get_loc(day)
        if i <= train_window_size:
            raise Exception('Not enough data on symbol {} to train model for day {}!'.format(self.symbol, day))
        X_train = self._features.iloc[i - train_window_size:i]
        y_train = self._target.iloc[i - train_window_size:i]
        close_train = self._close.iloc[i - train_window_size:i]
        X_test = self._features.iloc[i:i + 1]
        y_test = self._target.iloc[i:i + 1]
        close_test = self._close.iloc[i:i + 1]
        return X_train, y_train, close_train, X_test, y_test, close_test

    def get_train_data(self, day):
        X_train, y_train, close_train, X_test, y_test, close_test = self.get_data(day, self.window_size)
        return X_train, y_train, close_train

    def get_test_data(self, day):
        X_train, y_train, close_train, X_test, y_test, close_test = self.get_data(day, self.window_size)
        return X_test, y_test, close_test

    def train_estimator(self, day):
        X_train, y_train, close_train = self.get_train_data(day)
        est = self.pipeline.estimator
        if not self.parameters is None:
            est.set_params(**self.parameters)
        try:
            est = est.fit(X_train, y_train)
        except:
            # Fit failed
            raise Exception('Fit failed for symbol {} (Day {})!'.format(self.symbol, day))
        return est

    def get_estimator(self, day):
        # If no cache dir is set, skip caching
        if not self.cache_dir:
            return self.train_estimator(day)
        # Determine model cache file name and path
        cached_name = "model_{}.p".format(day.strftime("D%Y%m%dT%H%M%S"))
        cached_path = "{}/{}".format(self.cache_dir, cached_name)
        # If a cached model exists, return it
        if os.path.exists(cached_path):
            with open(cached_path, "rb") as f:
                return pickle.load(f)
        # Otherwise, train a new model
        else:
            est = self.train_estimator(day)
            # Save the trained model in cache directory
            with open(cached_path, "wb") as f:
                pickle.dump(est, f)
            # Finally, return the estimator
            return est

    def get_signal(self, day):
        est = self.get_estimator(day)
        features, label, close = self.get_test_data(day)
        # Reshape the input
        shaped = np.reshape(features.values, (1, -1))
        pred = est.predict(shaped)
        # Wrap the signals in a dataframe
        # result = pd.DataFrame()
        # result.loc[day, '{}'.format(self.symbol)] = pred[0]
        # result.loc[day, '{}_label'.format(self.symbol)] = label.iloc[0]
        # result.loc[day, '{}_close'.format(self.symbol)] = close.iloc[0]
        # return result
        res = Signal(day=day, symbol=self.symbol, signal=SignalType.parse(pred[0]), label=SignalType.parse(label.iloc[0]), close=float(close.iloc[0]))
        return res