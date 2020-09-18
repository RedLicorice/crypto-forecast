from sklearn.pipeline import Pipeline

# This class exposes coef_ and feature_importances_ from a pipeline in order to
#  use it for feature selection in wrapper methods such as RFECV or SelectFromModel
class SelectionPipeline(Pipeline):
    def fit(self, X, y=None, **fit_params):
        """Calls last elements .coef_ method.
        Based on the sourcecode for decision_function(X).
        Link: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/pipeline.py
        ----------
        """
        super(SelectionPipeline, self).fit(X, y, **fit_params)
        # We're assuming classifier is the last element of the pipeline
        clf = self.steps[-1][-1]
        if hasattr(clf, 'coef_'):
            self.coef_ = clf.coef_
        if hasattr(clf, 'feature_importances_'):
            self.feature_importances_ = clf.feature_importances_
        return self
