from enum import Enum
from lib.log import logger
import wrapt
import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning

## Decorator for attachable functionality in models
@wrapt.decorator
def with_x(wrapped, instance, args, kwargs):
    def _execute(*args, **kwargs):
        if kwargs.get('x_train') is None:
            raise ValueError('Missing x_train!')
        if kwargs.get('x_test') is None:
            raise ValueError('Missing x_test!')
        return wrapped(*args, **kwargs)
    return _execute(*args, **kwargs)

@wrapt.decorator
def with_xy(wrapped, instance, args, kwargs):
    def _execute(*args, **kwargs):
        if kwargs.get('x_train') is None:
            raise ValueError('Missing x_train!')
        if kwargs.get('y_train') is None:
            raise ValueError('Missing y_train!')
        if kwargs.get('x_test') is None:
            raise ValueError('Missing x_test!')
        if kwargs.get('y_test') is None:
            raise ValueError('Missing y_test!')
        return wrapped(*args, **kwargs)
    return _execute(*args, **kwargs)

@wrapt.decorator
def with_y(wrapped, instance, args, kwargs):
    def _execute(*args, **kwargs):
        if kwargs.get('y') is None:
            raise ValueError('Missing y!')
        return wrapped(*args, **kwargs)
    return _execute(*args, **kwargs)

@wrapt.decorator
def with_default_params(wrapped, instance, args, kwargs):
    def _execute(*args, **kwargs):
        if kwargs.get('params') is None:
            kwargs.update({'params':instance.default_params})
        return wrapped(*args, **kwargs)
    return _execute(*args, **kwargs)

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
        self.default_params = kwargs.get('params', self.default_params)

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

class SMModel(Model):
    def evaluate_config(self, cfg, **kwargs):
        x_train, x_test = cfg['x_train'], cfg['x_test']
        params = cfg['params']

        history = [x for x in x_train]
        predictions = []
        errors = []
        scores = []
        for i in range(len(x_test)):
            yhat = 0
            try:
                _left = len(x_test) - i
                model_fit = self.fit(history, params=params)
                if not model_fit:
                    return
                forecast, fcerr, conf_int = model_fit.forecast(steps=_left)
                yhat = float(forecast[0]) # Forecast next element of the test set
                if np.isnan(yhat):
                    yhat = 0
            except ValueError:
                errors.append('ValueError at step {} for config {}'.format(i, str(params)))
                pass
            except ConvergenceWarning:
                errors.append('ConvergenceWarning at step {} for config {}'.format(i, str(params)))
                pass
            except HessianInversionWarning:
                errors.append('HessianInversionWarning at step {} for config {}'.format(i, str(params)))
                pass
            except np.linalg.LinAlgError:
                errors.append('LinAlgError at step {} for config {}'.format(i, str(params)))
                pass
            finally:
                predictions.append(yhat) # add forecasted y to predictions
                history.append(x_test[i]) # Add an element from test set to history

        if not predictions:
            logger.error('No predictions made for config {}'.format(str(params)))
        else:
            #return x_test, np.array(predictions), str(self), str(params), errors, scores
            return {
                'y': x_test,
                'y_pred': np.array(predictions),
                'model': str(self),
                'params': params,
                'errors': errors,
                'scores': scores
            }

class SKModel(Model):
    def evaluate_config(self, cfg, **kwargs):
        x_train, x_test, y_train, y_test = cfg['x_train'], cfg['x_test'], cfg['y_train'], cfg['y_test']
        params = cfg['params']

        logger.info('Testing:' + str(params))
        history = [x for x in x_train.values], [y for y in y_train.values]
        predictions = []
        train_predictions = []
        errors = []
        scores = []
        for i in range(len(x_test)):
            # logger.info('grid_search {} {} {}/{}'.format(self.name, str(order), i+1, len(x_test)))
            yhat = 0
            train_yhat = 0
            try:
                _left = len(x_test) - i
                model_fit = self.fit(history[0], y=history[1], params=params)
                scores.append(model_fit.score(history[0], y=history[1]))
                # Save train set prediction
                forecast = model_fit.predict(history[0])
                train_yhat = float(forecast[0])  # Forecast next element of the test set
                if np.isnan(yhat):
                    train_yhat = 0
                # Save test set prediction
                forecast = model_fit.predict(x_test[i:])  # ToDO: Save model, use new model only if score increases
                yhat = float(forecast[0])  # Forecast next element of the test set
                if np.isnan(yhat):
                    yhat = 0
            except Exception as e:
                errors.append('Error at step {} for config {}: {}'.format(i, str(params), str(e)))
                pass
            finally:
                train_predictions.append(train_yhat)
                predictions.append(yhat)  # add forecasted y to predictions
                history[0].append(x_test.iloc[i].values)  # Add an element from test set to history
                history[1].append(y_test.iloc[i])  # Add an element from test set to history

        if not predictions:
            logger.error('No predictions made for config {}'.format(str(params)))
        else:
            return {
                'y': y_test,
                'y_pred': np.array(predictions),
                'y_train': y_train,
                'y_train_pred': np.array(train_predictions),
                'model': str(self),
                'params': params,
                'errors': errors,
                'scores': scores
            }

class KModel(Model):
    def evaluate_config(self, cfg, **kwargs):
        x_train, x_test, y_train, y_test = cfg['x_train'], cfg['x_test'], cfg['y_train'], cfg['y_test']
        params = cfg['params']

        logger.info('Testing:' + str(params))
        history = [x for x in x_train.values], [y for y in y_train.values]
        predictions = []
        train_predictions = []
        errors = []
        scores = []
        for i in range(len(x_test)):
            # logger.info('grid_search {} {} {}/{}'.format(self.name, str(order), i+1, len(x_test)))
            yhat = 0
            train_yhat = 0
            try:
                _left = len(x_test) - i
                model_fit = self.fit(history[0], y=history[1], params=params)
                scores.append(model_fit.score(history[0], y=history[1]))
                # Save train set prediction
                forecast = model_fit.predict(history[0])
                train_yhat = float(forecast[0])  # Forecast next element of the test set
                if np.isnan(yhat):
                    train_yhat = 0
                # Save test set prediction
                forecast = model_fit.predict(x_test[i:])  # ToDO: Save model, use new model only if score increases
                yhat = float(forecast[0])  # Forecast next element of the test set
                if np.isnan(yhat):
                    yhat = 0
            except Exception as e:
                errors.append('Error at step {} for config {}: {}'.format(i, str(params), str(e)))
                pass
            finally:
                train_predictions.append(train_yhat)
                predictions.append(yhat)  # add forecasted y to predictions
                history[0].append(x_test.iloc[i].values)  # Add an element from test set to history
                history[1].append(y_test.iloc[i])  # Add an element from test set to history

        if not predictions:
            logger.error('No predictions made for config {}'.format(str(params)))
        else:
            return {
                'y': y_test,
                'y_pred': np.array(predictions),
                'y_train': y_train,
                'y_train_pred': np.array(train_predictions),
                'model': str(self),
                'params': params,
                'errors': errors,
                'scores': scores
            }