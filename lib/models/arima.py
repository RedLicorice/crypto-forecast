from lib.models import SMModel, ModelType, ModelFactory, with_params, with_x
from lib.log import logger
from lib.utils import to_discrete_double
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
import numpy as np

class ARIMAModel(SMModel):
    type = [ModelType.CONTINUOUS_PRICE, ModelType.UNIVARIATE]
    name = 'statsmodels.arima'
    default_params = {
        'order': (1,1,1)
    }

    @with_params
    def fit(self, x, **kwargs):
        params = kwargs.get('params')
        try:
            self.model = ARIMA(x, order=params['order']) \
                    .fit(disp=params.get('disp',0))
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
            logger.error('{}: convergence error'.format(str(self.model.get_params())))

    @with_x
    def get_grid_search_configs(self, **kwargs):
        x_train = kwargs.get('x_train')
        x_test = kwargs.get('x_test')

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
                        'params':{'order': (p,d,q)},
                        'x_train' : x_train,
                        'x_test' : x_test
                    })
        return configs

ModelFactory.register_model('arima', ARIMAModel)
