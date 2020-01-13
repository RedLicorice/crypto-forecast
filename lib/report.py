import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error

class Report:

    def __init__(self, labels: pd.DataFrame, prediction: np.ndarray, **kwargs):
        self.labels = labels.copy()
        self.predictions = prediction.copy()
        self.classifier = kwargs.get('classifier')
        self.params = kwargs.get('parameters')
        self.errors = kwargs.get('errors')
        self.comparison = kwargs.get('comparison', 'mse')

    # For grid search
    def __lt__(self, other):
        if self.comparison == 'mse':
            return self.mse() < other.mse()
        elif self.comparison == 'accuracy':
            return self.accuracy() < other.accuracy()

    def __repr__(self):
        return "{}({})".format(self.classifier, self.params)

    def accuracy(self):
        return accuracy_score(self.labels, self.predictions)

    def mse(self):
        return mean_squared_error(self.labels, self.predictions)