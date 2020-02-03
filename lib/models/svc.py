from lib.models import *
from lib.log import logger
from sklearn.svm import SVC
import numpy as np


class SVCModel(SKModel):
    type = [ModelType.DISCRETE, ModelType.MULTIVARIATE]
    name = 'sklearn.svc'
    default_params = {
        'kernel':'rbf',
        'C':1.0,
        'gamma':'auto',
    }

    @with_y
    @with_params
    def fit(self, x, **kwargs):
        y = kwargs.get('y')
        params = kwargs.get('params')
        self.model = SVC(**params)
        self.model.fit(x, y)
        return self.model

    def predict(self, x, **kwargs):
        pred = self.model.predict(x)
        return pred

    @with_xy
    def get_grid_search_configs(self, **kwargs):
        kernels = ['rbf']
        cs = np.arange(0.5, 5.0, 0.5)
        gammas = ['auto', 0.001, 0.005, 0.01, 0.05]

        # Get all possible configs
        configs = []
        for k in kernels:
            for c in cs:
                for g in gammas:
                    configs.append({
                        'params':{
                            'kernel': k,
                            'C': c,
                            'gamma':g,
                        },
                        'x_train' : kwargs.get('x_train'),
                        'y_train' : kwargs.get('y_train'),
                        'x_test' : kwargs.get('x_test'),
                        'y_test' : kwargs.get('y_test')
                    })
        return configs