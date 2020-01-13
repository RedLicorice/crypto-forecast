from sklearn.model_selection import train_test_split
from .symbol import Symbol, DatasetType
from .models import Model, ModelType, ARIMAModel
from .report import Report
from multiprocessing import Pool, cpu_count

class Job:
    def __init__(self, symbol: Symbol, model: Model):
        self.symbol = symbol
        self.model = model

    def __repr__(self):
        return 'job_' + str(self.symbol) + '_' + str(self.model)

    def get_dataset(self, **kwargs):
        x, y = None, None
        if self.model.is_type(ModelType.CONTINUOUS_PCT):
            x, y =  self.symbol.get_xy(DatasetType.CONTINUOUS_TA)

        elif self.model.is_type(ModelType.CONTINUOUS_PRICE):
            x, y = self.symbol.get_xy(DatasetType.OHLCV)

        elif self.model.is_type(ModelType.CONTINUOUS_PRICE_PCT):
            x, y =  self.symbol.get_dataset(DatasetType.OHLCV_PCT)

        elif self.model.is_type(ModelType.DISCRETE):
            x, y = self.symbol.get_dataset(DatasetType.DISCRETE_TA)

        if self.model.is_type(ModelType.UNIVARIATE):
            x = x[kwargs.get('univariate_target','close')]

        if kwargs.get('range'):
            r = kwargs.get('range')
            x = x.loc[r[0]:r[1]]
            y = y.loc[r[0]:r[1]]

        return x, y

    def holdout(self, **kwargs):
        x, y = self.get_dataset(range=kwargs.get('range'))

        # Won't shuffle by default
        x_train, x_test, y_train, y_test  = train_test_split(x, y,
                                                             train_size=kwargs.get('train_size', 0.7),
                                                             shuffle=kwargs.get('shuffle',False))
        if kwargs.get('validate', False):
            x_test, x_val, y_test, y_val = train_test_split(x, y, # Data is already shuffled if need be
                                                             test_size=kwargs.get('validation_size', 0.2))
        # Fit the model (on the train set) and make a prediction (on the test set)
        if self.model.fit(x_train.values, y_train.values):
            pred = self.model.predict(x_test.values)
            # Validate the model if needed
            # Plot results
            if pred:
                print("Fit Success! " + str(self.model.params['order']))
                return Report(y=y_test, prediction=pred, model=self.model)

    def expanding_window(self):
        pass

    def grid_search(self, **kwargs):
        x, y = self.get_dataset(range=kwargs.get('range'))

        # Won't shuffle by default
        x_train, x_test, y_train, y_test  = train_test_split(x, y,
                                                             train_size=kwargs.get('train_size', 0.7),
                                                             shuffle=kwargs.get('shuffle',False))

        configs = self.model.get_grid_search_configs(x_train=x_train, x_test=x_test)
        configs = configs[:8]
        # Parallelize
        if kwargs.get('multiprocessing', False):
            with Pool(cpu_count()) as pool:
                results = pool.map(self.model.evaluate_config, configs)
        else:
            results = []
            for cfg in configs:
                results.append(self.model.evaluate_config(cfg))
        #x_test, np.array(predictions), str(self), str(order), errors
        reports = [Report(labels=r[0], prediction=r[1], classifier=r[2], parameters=r[3], errors=r[4])
                    for r in results if r is not None]
        return reports