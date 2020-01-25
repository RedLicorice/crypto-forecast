from sklearn.neighbors import KNeighborsClassifier
from lib.models import *
from lib.log import logger
import numpy as np


class KNNModel(SKModel):
    type = [ModelType.DISCRETE, ModelType.MULTIVARIATE]
    name = 'sklearn.knn'
    default_params = {
        'n_neighbors':5,
        'weights':'uniform',
        'leaf_size':30,
        'p':2,
        'n_jobs':None
    }

    @with_y
    @with_default_params
    def fit(self, x, **kwargs):
        y = kwargs.get('y')
        params = kwargs.get('params')
        self.model = KNeighborsClassifier(**params)
        self.model.fit(x, y)
        return self.model

    def predict(self, x, **kwargs):
        pred = self.model.predict(x)
        return pred

    @with_xy
    def get_grid_search_configs(self, **kwargs):
        weights = ['uniform','distance']
        n_neighbors = [2, 3, 5, 9]
        leaf_size = [10]
        P = [2]

        # Get all possible configs
        configs = []
        for w in weights:
            for n in n_neighbors:
                for l in leaf_size:
                    for p in P:
                        configs.append({
                            'params':{
                                'n_neighbors':n,
                                'weights':w,
                                'leaf_size':l,
                                'p':p,
                                'n_jobs':None
                            },
                            'x_train' : kwargs.get('x_train'),
                            'y_train' : kwargs.get('y_train'),
                            'x_test' : kwargs.get('x_test'),
                            'y_test' : kwargs.get('y_test')
                        })
        return configs