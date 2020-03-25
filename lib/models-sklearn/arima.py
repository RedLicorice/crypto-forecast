import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

from lib.log import logger
from lib.utils import to_discrete_double

class ArimaEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param
        self._model = None

    def fit(self, X, y):
        params = self.get_params()
        order = (params['p'], params['d'], params['q'])
        try:
            self._model = ARIMA(X, order=order) \
                    .fit(disp=params.get('disp',0))
            return self
        except (ValueError, np.linalg.linalg.LinAlgError):
            logger.error('fit: ARIMA convergence error (order {} {} {})'.format(params['p'], params['d'], params['q']))
            return None

    def predict(self, X):
        params = self.get_params()

        # Check is fit had been called
        check_is_fitted(self)

        if not self._model:
            return None
        try:
            forecast = self._model.forecast(steps=X.shape[0])
            self.y_ = to_discrete_double(forecast, -0.01, 0.01)
        except (ValueError, np.linalg.linalg.LinAlgError):
            logger.error('predict: ARIMA convergence error (order {} {} {})'.format(params['p'], params['d'], params['q']))