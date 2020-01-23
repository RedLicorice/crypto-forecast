from sklearn.model_selection import train_test_split
from .symbol import Symbol, DatasetType
from .models import Model, ModelType
from .report import Report
from .log import logger
from .utils import oversample, undersample
from multiprocessing import Pool, cpu_count

class Job:
    def __init__(self, symbol: Symbol, model: Model):
        self.symbol = symbol
        self.model = model

    def __repr__(self):
        return 'job_' + str(self.symbol) + '_' + str(self.model)

    def get_dataset(self, **kwargs):
        x, y = None, None
        x_type = kwargs.get('dataset')
        y_type = kwargs.get('target')
        if not x_type:
            if self.model.is_type(ModelType.CONTINUOUS_PCT):
                x_type = DatasetType.CONTINUOUS_TA
            elif self.model.is_type(ModelType.CONTINUOUS_PRICE):
                x_type = DatasetType.OHLCV
            elif self.model.is_type(ModelType.CONTINUOUS_PRICE_PCT):
                x_type = DatasetType.OHLCV_PCT
            elif self.model.is_type(ModelType.DISCRETE):
                x_type = DatasetType.DISCRETE_TA
        if not y_type:
            if self.model.is_type(ModelType.CONTINUOUS_PCT):
                y_type = DatasetType.CONTINUOUS_TA
            elif self.model.is_type(ModelType.CONTINUOUS_PRICE):
                y_type = DatasetType.OHLCV
            elif self.model.is_type(ModelType.CONTINUOUS_PRICE_PCT):
                y_type = DatasetType.OHLCV_PCT
            elif self.model.is_type(ModelType.DISCRETE):
                y_type = DatasetType.DISCRETE_TA

        x, y = self.symbol.get_xy(x_type)
        if y_type != x_type:
            _x, y = self.symbol.get_xy(y_type)
        # if kwargs.get('dataset'):
        #     x, y = self.symbol.get_xy(kwargs.get('dataset'))
        # else:
        #     if self.model.is_type(ModelType.CONTINUOUS_PCT):
        #         x, y =  self.symbol.get_xy(DatasetType.CONTINUOUS_TA)
        #
        #     elif self.model.is_type(ModelType.CONTINUOUS_PRICE):
        #         x, y = self.symbol.get_xy(DatasetType.OHLCV)
        #
        #     elif self.model.is_type(ModelType.CONTINUOUS_PRICE_PCT):
        #         x, y =  self.symbol.get_xy(DatasetType.OHLCV_PCT)
        #
        #     elif self.model.is_type(ModelType.DISCRETE):
        #         x, y = self.symbol.get_xy(DatasetType.DISCRETE_TA)

        if self.model.is_type(ModelType.UNIVARIATE):
            univar_col = kwargs.get('univariate_column')
            x = x[univar_col if univar_col else 'close']

        logger.info("|x|={},{} |y|={},{}".format(x.shape[0], x.shape[1] if len(x.shape) > 1 else 0, y.shape[0], y.shape[1] if len(y.shape) > 1 else 0))
        return x, y

    def holdout(self, **kwargs):
        x, y = self.get_dataset(dataset=kwargs.get('dataset'), target=kwargs.get('target'), univariate_column=kwargs.get('univariate_column'))
        params = kwargs.get('params')
        # Won't shuffle by default
        x_train, x_test, y_train, y_test  = train_test_split(x, y,
                                                             train_size=kwargs.get('train_size', 0.7),
                                                             shuffle=kwargs.get('shuffle',False))
        if kwargs.get('validate', False):
            x_test, x_val, y_test, y_val = train_test_split(x, y, # Data is already shuffled if need be
                                                             test_size=kwargs.get('validation_size', 0.2))
        # Apply oversampling/undersampling if needed
        if kwargs.get('oversample'):
            x_train, y_train = oversample(x_train, y_train)
        elif kwargs.get('undersample'):
            x_train, y_train = undersample(x_train, y_train)

        # Fit the model (on the train set) and make a prediction (on the test set)
        if self.model.fit(x_train.values, y=y_train.values, params=params):
            pred = self.model.predict(x_test.values)
            # Validate the model if needed
            # Plot results
            if pred is not None:
                #print("Fit Success! " + str(self.model.params['order']))
                return Report(**{
                        'y': y_test,
                        'y_pred': pred,
                        'model': str(self.model),
                        'params': params,
                        'errors': [],
                        'scores': [],
                        'symbol': self.symbol
                    })

    def expanding_window(self):
        pass

    def grid_search(self, **kwargs):
        x, y = self.get_dataset(dataset=kwargs.get('dataset'), univariate_column=kwargs.get('univariate_column'))

        # Won't shuffle by default
        x_train, x_test, y_train, y_test  = train_test_split(x, y,
                                                             train_size=kwargs.get('train_size', 0.7),
                                                             shuffle=kwargs.get('shuffle',False))
        # Apply oversampling/undersampling if needed
        if kwargs.get('oversample'):
            x_train, y_train = oversample(x_train, y_train)
        elif kwargs.get('undersample'):
            x_train, y_train = undersample(x_train, y_train)

        configs = self.model.get_grid_search_configs(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        #configs = configs[:8]
        # Parallelize
        if kwargs.get('multiprocessing', False):
            with Pool(cpu_count()) as pool:
                results = pool.map(self.model.evaluate_config, configs)
        else:
            results = []
            for cfg in configs:
                results.append(self.model.evaluate_config(cfg))
        #x_test, np.array(predictions), str(self), str(order), errors
        reports = [Report(**{**r , **{'symbol':self.symbol}}) for r in results if r is not None]
        return reports