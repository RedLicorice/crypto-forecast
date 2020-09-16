from imblearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

PARAMETER_GRID = {
    'c__C': [0.1, 0.5, 1, 1.5, 2],
    'c__kernel': ['rbf'], #'linear', 'rbf''linear', 'poly'
    'c__gamma': ['auto'],
    #'c__base_estimator__degree': [2, 3, 4, 5, 10], # only makes sense for poly
    'c__coef0': [0.0, 0.1, 0.2, 0.4, 0.5],
}

estimator = Pipeline([
    ('i', SimpleImputer()), # Replace nan's with the median value between previous and next observation
    ('s', RobustScaler()), # Scale data in order to center it and increase robustness against noise and outliers
    ('c', SVC()),
])