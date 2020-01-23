from lib.models import *
from lib.log import logger
from lib.utils import from_categorical, to_categorical
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
import numpy as np

class NNModel(KModel):
    type = [ModelType.DISCRETE]
    name = 'keras.nn'
    default_params = {
        'neurons':[32],
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100
    }
    metrics = ['acc', 'mse', 'mae', 'mape']

    @with_y
    @with_default_params
    def fit(self, x, **kwargs):
        y = kwargs.get('y')
        params = kwargs.get('params')
        if y is None:
            raise RuntimeError('Missing y (labels) parameters!')

        # one hot encode output
        y = to_categorical(y)
        self.model = Sequential()
        for n in params['neurons']:
            self.model.add(Dense(
                n,
                input_dim=x.shape[1],
                activation='tanh',
                # kernel_regularizer=L1L2(l1=0.5, l2=0.1)
            ))

        self.model.add(Dense(
            3,  # output dim is 3, one score per each class
            activation='softmax',
            # kernel_regularizer=L1L2(l1=0.2, l2=0.4),
            input_dim=x.shape[1]  # input dimension = number of features
        ))
        optimizer = Adam(lr=params['learning_rate'])
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=self.metrics)

        if self.model is None:
            raise RuntimeError("No model compiled!")
        self.model.fit(x, y, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        return self.model

    def predict(self, x, **kwargs):
        pred = self.model.predict(x)
        return from_categorical(pred)

    @with_xy
    def get_grid_search_configs(self, **kwargs):
        x_train = kwargs.get('x_train')
        x_test = kwargs.get('x_test')
        y_train = kwargs.get('y_train')
        y_test = kwargs.get('y_test')

        neurons = [[8], [16], [24], [32], [64]]
        learning_rates = np.logspace(0.001, 0.01, 5)
        batch_sizes = [8, 16, 32, 64]
        epochs = [100, 200, 300]

        # Get all possible configs
        configs = []
        for n in neurons:
            for lr in learning_rates:
                for b in batch_sizes:
                    for e in epochs:
                        configs.append({
                            'params':{
                                'neurons':n,
                                'learning_rate': lr,
                                'batch_size': b,
                                'epochs': e
                            },
                            'x_train' : x_train,
                            'y_train' : y_train,
                            'x_test' : x_test,
                            'y_test' : y_test
                        })
        return configs