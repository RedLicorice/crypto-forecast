# Just a subclass of BaggingClassifier with added coefs for use with RFECV and Sklearn-Genetic
import numpy as np
from sklearn.ensemble import BaggingClassifier


class BaggingClassifierCoefs(BaggingClassifier):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # add attribute of interest
        self.feature_importances_ = None

    def fit(self, X, y, sample_weight=None):
        # overload fit function to comute feature_importance
        fitted = self._fit(X, y, self.max_samples, sample_weight=sample_weight) # hidden fit function
        if hasattr(fitted.estimators_[0], 'feature_importances_'):
            self.feature_importances_ =  np.mean([tree.feature_importances_ for tree in fitted.estimators_], axis=0)
        else:
            self.feature_importances_ =  np.mean([tree.coef_ for tree in fitted.estimators_], axis=0)
        return(fitted)