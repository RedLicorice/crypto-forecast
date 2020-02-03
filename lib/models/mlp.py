from lib.models import *
from lib.log import logger
from sklearn.neural_network import MLPClassifier
import numpy as np

class MLPModel(SKModel):
    type = [ModelType.DISCRETE]
    name = 'sklearn.mlp'
    default_params = {
        'hidden_layer_sizes': (100,),
        'solver': 'adam',
        'learning_rate':'constant',
        'learning_rate_init':0.001,
        'activation':'relu'
    }

    @with_y
    @with_params
    def fit(self,x, **kwargs):
        y = kwargs.get('y')
        params = kwargs.get('params')

        if y is None:
            raise RuntimeError('Missing y (labels) parameters!')
        self.model = MLPClassifier(**params)
        self.model.fit(x, y)
        return self.model

    def predict(self, x, **kwargs):
        pred = self.model.predict(x)
        return pred

    @with_xy
    def get_grid_search_configs(self, **kwargs):
        x_train = kwargs.get('x_train')
        x_test = kwargs.get('x_test')
        y_train = kwargs.get('y_train')
        y_test = kwargs.get('y_test')
        if x_train is None or x_test is None or y_train is None or y_test is None:
            raise ValueError('Missing required x_train and x_test parameters!')

        layer_size = [(8,), (10,), (50,), (10,4), (8,3)]
        solvers = ['adam', 'lbfgs']
        learning_rate = ['constant', 'invscaling', 'adaptive']
        learning_rate_init = [0.001, 0.005, 0.01, 0.05]
        act = ['relu','logistic','tanh']

        # Get all possible configs
        configs = []
        for l in layer_size:
            for s in solvers:
                for lr in learning_rate:
                    for lri in learning_rate_init:
                        for a in act:
                            configs.append({
                                'params': {
                                    'hidden_layer_sizes': l,
                                    'solver': s,
                                    'learning_rate': lr,
                                    'learning_rate_init': lri,
                                    'activation': a
                                },
                                'x_train': x_train,
                                'y_train': y_train,
                                'x_test': x_test,
                                'y_test': y_test
                            })
        return configs