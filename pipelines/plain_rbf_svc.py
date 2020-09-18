from imblearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer


TARGET='binary_bin'

PARAMETER_GRID = {
    'c__C': [0.1, 0.5, 1, 1.5, 2], # Regularization parameter. The strength of the regularization is inversely proportional to C. >0
    'c__kernel': ['rbf'],
    'c__gamma': ['scale', 'auto'], # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. (default = 'scale')
    'c__class_weight':[None, 'balanced']
}

estimator = Pipeline([
    ('i', SimpleImputer()), # Replace nan's with the median value between previous and next observation
    ('s', RobustScaler()), # Scale data in order to center it and increase robustness against noise and outliers
    ('c', SVC()),
])