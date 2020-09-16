from lib.models import SKModel, ModelType, ModelFactory, with_params, with_y, with_xy
from old.lib.utils import scale, has_negative
from sklearn.naive_bayes import MultinomialNB


class MNBModel(SKModel):
    type = [ModelType.DISCRETE, ModelType.MULTIVARIATE]
    name = 'sklearn.mnb'
    default_params = {
        'alpha': 1.0,
        'fit_prior':True,
    }

    @with_y
    @with_params
    def fit(self, x, **kwargs):
        y = kwargs.get('y')
        params = kwargs.get('params')
        if has_negative(x):
            x = scale(x, scaler='minmax', feature_range=(0,1))
        if has_negative(y):
            y = scale(y, scaler='minmax', feature_range=(0,1))
        self.model = MultinomialNB(**params)
        self.model.fit(x, y)
        return self.model

    def predict(self, x, **kwargs):
        if has_negative(x):
            x = scale(x, scaler='minmax', feature_range=(0,1))
        pred = self.model.predict(x)
        return pred

    @with_xy
    def get_grid_search_configs(self, **kwargs):
        alpha = [0.0000000001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
        fit_prior = [True, False]
        # Get all possible configs
        configs = []
        for k in alpha:
            for f in fit_prior:
                configs.append({
                    'params':{
                        'alpha': k,
                        'fit_prior': f
                    },
                    'x_train' : kwargs.get('x_train'),
                    'y_train' : kwargs.get('y_train'),
                    'x_test' : kwargs.get('x_test'),
                    'y_test' : kwargs.get('y_test')
                })
        return configs

ModelFactory.register_model('mnb', MNBModel)