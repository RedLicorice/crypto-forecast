from sklearn.svm import SVC
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning
from .log import logger
from .utils import to_discrete_double
import numpy as np
from enum import Enum

class ModelType(Enum):
    CONTINUOUS_PCT = 1
    CONTINUOUS_PRICE = 2
    CONTINUOUS_PRICE_PCT = 3
    DISCRETE = 4
    UNIVARIATE = 5
    MULTIVARIATE = 6

def create_model(_cls, params):
    return _cls(params=params)

class Model:
    type = [ModelType.CONTINUOUS_PCT, ModelType.MULTIVARIATE]
    name = 'model'
    default_params = {}
    model = None

    def __init__(self, **kwargs):
        self.params = kwargs.get('params', self.default_params)

    def __repr__(self):
        return self.name

    def fit(self, x, **kwargs):
        pass

    def predict(self, x, **kwargs):
        pass

    def is_type(self, type):
        return type in self.type

    def get_grid_search_configs(self, **kwargs):
        pass

    def evaluate_config(self, cfg, **kwargs):
        pass

class SVCModel(Model):
    type = [ModelType.DISCRETE, ModelType.MULTIVARIATE]
    name = 'sklearn.svc'
    default_params = {
        'kernel':'rbf',
        'C':1.0
    }

    def get_grid_search_configs(self, **kwargs):
        x_train = kwargs.get('x_train')
        x_test = kwargs.get('x_test')
        y_train = kwargs.get('y_train')
        y_test = kwargs.get('y_test')
        if x_train is None or x_test is None or y_train is None or y_test is None:
            raise ValueError('Missing required x_train and x_test parameters!')

        kernels = ['rbf']
        cs = np.arange(0.5, 5.0, 0.5)

        # Get all possible configs
        configs = []
        for k in kernels:
            for c in cs:
                configs.append({
                    'kernel': k,
                    'c': c,
                    'train' : x_train,
                    'y_train' : y_train,
                    'test' : x_test,
                    'y_test' : y_test
                })
        return configs

    def evaluate_config(self, cfg, **kwargs):
        x_train, x_test, y_train, y_test= cfg['train'], cfg['test'], cfg['y_train'], cfg['y_test']
        k, c = cfg['kernel'], cfg['c']
        _cfg = 'kernel={}, c={}'.format(k, c)
        history = [x for x in x_train.values],[y for y in y_train.values]

        predictions = []
        errors = []
        for i in range(len(x_test)):
            # logger.info('grid_search {} {} {}/{}'.format(self.name, str(order), i+1, len(x_test)))
            yhat = 0
            try:
                _left = len(x_test) - i
                model_fit = self.fit(history[0], y=history[1], kernel=k, c=c)
                forecast = model_fit.predict(x_test[i:])
                yhat = float(forecast[0]) # Forecast next element of the test set
                if np.isnan(yhat):
                    yhat = 0
            except Exception as e:
                errors.append('Error at step {} for config {}: {}'.format(i, _cfg, str(e)))
                pass
            finally:
                predictions.append(yhat) # add forecasted y to predictions
                history[0].append(x_test.iloc[i]) # Add an element from test set to history
                history[1].append(y_test.iloc[i]) # Add an element from test set to history

        if not predictions:
            logger.error('No predictions made for config {}'.format(_cfg))
        else:
            return y_test, np.array(predictions), str(self), _cfg, errors

    def fit(self, x, **kwargs):
        y = kwargs.get('y')
        if y is None:
            raise RuntimeError('Missing y (labels) parameters!')
        self.model = SVC(**self.params)
        self.model.fit(x, y)
        return self.model

    def predict(self, x, **kwargs):
        pred = self.model.predict(x)
        return pred

class ExpSmoothModel(Model):
    type = [ModelType.CONTINUOUS_PRICE, ModelType.UNIVARIATE]
    name = 'expsmooth'
    default_params = {
        'seasonal_periods':5,
        'alpha': 0.5,
        'beta':0.5,
    }

    def get_grid_search_configs(self, **kwargs):
        x_train = kwargs.get('x_train')
        x_test = kwargs.get('x_test')
        if x_train is None or x_test is None:
            raise ValueError('Missing required x_train and x_test parameters!')
        seasonal_periods = range(1,14)
        alpha = np.arange(0.5, 3, 0.5)
        beta = np.arange(0.0, 3.0, 0.5)

        # Get all possible configs
        configs = []
        for p in seasonal_periods:
            for a in alpha:
                for b in beta:
                    configs.append({
                        'seasonal_periods': p,
                        'alpha': a,
                        'beta': b,
                        'train' : x_train,
                        'test' : x_test
                    })
        return configs

    def evaluate_config(self, cfg, **kwargs):
        x_train, x_test = cfg['train'], cfg['test']
        s, a, b = cfg['seasonal_periods'], cfg['alpha'], cfg['beta']
        _cfg = 'seasonal_periods={}, alpha={}, beta={}'.format(s,a,b)
        history = [x for x in x_train]
        predictions = []
        errors = []
        for i in range(len(x_test)):
            # logger.info('grid_search {} {} {}/{}'.format(self.name, str(order), i+1, len(x_test)))
            yhat = 0
            try:
                _left = len(x_test) - i
                model_fit = self.fit(np.array(history), params={'alpha':a, 'beta':b, 'seasonal_periods':s})
                if not model_fit:
                    return
                forecast, fcerr, conf_int = model_fit.forecast(steps=_left) # forecast only returns one step
                yhat = float(forecast) # Forecast next element of the test set
                if np.isnan(yhat):
                    yhat = 0
            except ValueError:
                errors.append('ValueError at step {} for config {}'.format(i, _cfg))
                pass
            except ConvergenceWarning:
                errors.append('ConvergenceWarning at step {} for config {}'.format(i, _cfg))
                pass
            except HessianInversionWarning:
                errors.append('HessianInversionWarning at step {} for config {}'.format(i, _cfg))
                pass
            except np.linalg.LinAlgError:
                errors.append('LinAlgError at step {} for config {}'.format(i, _cfg))
                pass
            finally:
                predictions.append(yhat) # add forecasted y to predictions
                history.append(x_test[i]) # Add an element from test set to history

        if not predictions:
            logger.error('No predictions made for config {}'.format(_cfg))
        else:
            return x_test, np.array(predictions), str(self), _cfg, errors

    def fit(self, x, **kwargs):
        params = kwargs.get('params',self.params)
        try:
            if params.get('beta'):
                # Use Augmented Dickey-Fuller test: try to reject the null hypothesis that the series
                # has a unit root, i.e. it's stationary. Statsmodels implementation has issues with window
                # size lower than 6, so adapt it for now.
                # Using the p-value, check if it's possible to reject the null hypothesis at 5 % significance
                # level and than choose Simple Exponential Smoothing or Holt's Exponential Smoothing (additional
                # trend factor).
                r = adfuller(x) if x.shape[0] > 6 else adfuller(x, maxlag=4)
                pvalue = r[1]
                if pvalue < 0.05:
                    self.model = ExponentialSmoothing(x, trend=None, seasonal=None) \
                        .fit(smoothing_level=params.get('alpha'))
                else:
                    self.model = ExponentialSmoothing(x, trend='additive', seasonal='additive',
                                                     seasonal_periods=params.get('seasonal_periods')) \
                        .fit(smoothing_level=params.get('alpha'), smoothing_slope=self.params.get('beta'))
            else:
                self.model = ExponentialSmoothing(x, trend=None, seasonal=None) \
                    .fit(smoothing_level=params.get('alpha'))
            self.params = params
            return self.model
        except (ValueError, np.linalg.linalg.LinAlgError):
            logger.error('Exponential Smoothing convergence error (a:{},b:{})'.format(params.get('alpha'), params.get('beta')))
            return None

    def predict(self, x, **kwargs):
        try:
            forecast = self.model.forecast(steps=x.shape[0])
            return to_discrete_double(forecast, -0.01, 0.01)
        except (ValueError, np.linalg.linalg.LinAlgError):
            logger.error('Exponential Smoothing convergence error (a:{},b:{})'.format(self.params.get('alpha'), self.params.get('beta')))
            return

class ARIMAModel(Model):
    type = [ModelType.CONTINUOUS_PRICE, ModelType.UNIVARIATE]
    name = 'arima'
    default_params = {
        'order': (1,1,1)
    }

    def get_grid_search_configs(self, **kwargs):
        x_train = kwargs.get('x_train')
        x_test = kwargs.get('x_test')
        if x_train is None or x_test is None:
            raise ValueError('Missing required x_train and x_test parameters!')
        p_values = range(0,6)
        d_values = range(0,6)
        q_values = range(0,6)
        # If series is stationary, don't apply differentiation
        adf = adfuller(x_train)  # 0 is score, 1 is pvalue
        if adf[1] < 0.05:  # Null hp rejected, series is stationary and requires no differentiation
            logger.info('Series is stationary, no need for differencing')
            d_values = [0] # Set d = 0
        # Get all possible configs
        configs = []
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    configs.append({
                        'order': (p,d,q),
                        'train' : x_train,
                        'test' : x_test
                    })
        return configs

    def evaluate_config(self, cfg, **kwargs):
        x_train, x_test, order = cfg['train'], cfg['test'], cfg['order']
        _cfg = 'p={}, d={}, q={}'.format(order[0],order[1],order[2])
        history = [x for x in x_train]
        predictions = []
        errors = []
        for i in range(len(x_test)):
            # logger.info('grid_search {} {} {}/{}'.format(self.name, str(order), i+1, len(x_test)))
            yhat = 0
            try:
                _left = len(x_test) - i
                model_fit = self.fit(history, params={'order':order})
                if not model_fit:
                    return
                forecast, fcerr, conf_int = model_fit.forecast(steps=_left)
                yhat = float(forecast[0]) # Forecast next element of the test set
                if np.isnan(yhat):
                    yhat = 0
            except ValueError:
                errors.append('ValueError at step {} for config {}'.format(i, _cfg))
                pass
            except ConvergenceWarning:
                errors.append('ConvergenceWarning at step {} for config {}'.format(i, _cfg))
                pass
            except HessianInversionWarning:
                errors.append('HessianInversionWarning at step {} for config {}'.format(i, _cfg))
                pass
            except np.linalg.LinAlgError:
                errors.append('LinAlgError at step {} for config {}'.format(i, _cfg))
                pass
            finally:
                predictions.append(yhat) # add forecasted y to predictions
                history.append(x_test[i]) # Add an element from test set to history

        if not predictions:
            logger.error('No predictions made for config {}'.format(_cfg))
        else:
            return x_test, np.array(predictions), str(self), _cfg, errors

    def fit(self, x, **kwargs):
        params = kwargs.get('params', self.params)
        try:
            self.model = ARIMA(x, order=params['order']) \
                    .fit(disp=params.get('disp',0))
            self.params = params
            return self.model
        except (ValueError, np.linalg.linalg.LinAlgError):
            logger.error('ARIMA convergence error (order {} {} {})'.format(params['order'][0], params['order'][1], params['order'][2]))
            return None

    def predict(self, x, **kwargs):
        if not self.model:
            return None
        try:
            forecast = self.model.forecast(steps=x.shape[0])
            return to_discrete_double(forecast, -0.01, 0.01)
        except (ValueError, np.linalg.linalg.LinAlgError):
            logger.error('ARIMA convergence error (order {} {} {})'.format(self.params['order'][0], self.params['order'][1], self.params['order'][2]))

        

