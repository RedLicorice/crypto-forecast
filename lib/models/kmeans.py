from lib.models import *
from lib.log import logger
from sklearn.cluster import KMeans
import numpy as np


class KMeansModel(SKModel):
    type = [ModelType.DISCRETE, ModelType.MULTIVARIATE]
    name = 'sklearn.kmeans'
    default_params = {
        'n_clusters':8,
        'init':'k-means++', # random or ndarray
        'n_init': 10,
        'max_iter':300, #Maximum number of iterations of the k-means algorithm for a single run
        'tol':0.00001, #Relative tolerance with regards to inertia to declare convergence
        'n_jobs':None,
    }

    @with_y
    @with_params
    def fit(self, x, **kwargs):
        y = kwargs.get('y')
        params = kwargs.get('params')
        self.model = KMeans(**params)
        self.model.fit(x, y)
        return self.model

    def predict(self, x, **kwargs):
        pred = self.model.predict(x)
        return pred

    @with_xy
    def get_grid_search_configs(self, **kwargs):
        # The number of clusters to form as well as the number of centroids to generate.
        n_clusters = [3, 5]
        # Number of time the k-means algorithm will be run with different centroid seeds.
        # The final results will be the best output of n_init consecutive runs in terms of inertia.
        n_init = [10, 15, 20]
        # Maximum number of iterations of the k-means algorithm for a single run
        max_iter = [150, 300, 600]
        # Relative tolerance with regards to inertia to declare convergence
        tolerance = [0.0001, 0.0005, 0.0010]

        # Get all possible configs
        configs = []
        for c in n_clusters:
            for n in n_init:
                for i in max_iter:
                    for t in tolerance:
                        configs.append({
                            'params':{
                                'n_clusters':c,
                                'init':'k-means++', # k-means++, random or ndarray
                                'n_init': n,
                                'max_iter':i,
                                'tol':t,
                                'n_jobs':None,
                            },
                            'x_train' : kwargs.get('x_train'),
                            'y_train' : kwargs.get('y_train'),
                            'x_test' : kwargs.get('x_test'),
                            'y_test' : kwargs.get('y_test')
                        })
        return configs