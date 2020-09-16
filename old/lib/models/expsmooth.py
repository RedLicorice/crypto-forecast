from lib.models import SMModel, ModelType, ModelFactory, with_params, with_x
from lib.log import logger
from old.lib.utils import to_discrete_double
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

class ExpSmoothModel(SMModel):
    type = [ModelType.CONTINUOUS_PRICE, ModelType.UNIVARIATE]
    name = 'statsmodels.expsmooth'
    default_params = {
        'seasonal_periods':5,
        'alpha': 0.5,
        'beta':0.5,
    }

    @with_params
    def fit(self, x, **kwargs):
        params = kwargs.get('params')
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

    @with_x
    def get_grid_search_configs(self, **kwargs):
        x_train = kwargs.get('x_train')
        x_test = kwargs.get('x_test')

        seasonal_periods = range(1,14)
        alpha = np.arange(0.5, 3, 0.5)
        beta = np.arange(0.0, 3.0, 0.5)

        # Get all possible configs
        configs = []
        for p in seasonal_periods:
            for a in alpha:
                for b in beta:
                    configs.append({
                        'params':{
                            'seasonal_periods': p,
                            'alpha': a,
                            'beta': b,
                        },
                        'x_train' : x_train,
                        'x_test' : x_test
                    })
        return configs

ModelFactory.register_model('expsmooth', ExpSmoothModel)