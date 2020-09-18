from imblearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer


TARGET='binary_bin'

PARAMETER_GRID = {
    'c__C': [0.1, 0.5, 1, 1.5, 2],
    # Regularization parameter. The strength of the regularization is inversely proportional to C. >0
    'c__kernel': ['poly'],
    'c__gamma': ['scale', 'auto'],
    # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. (default = 'scale')
    'c__degree': [2, 3, 4, 5],
    # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
    'c__coef0': [0.0, 0.1, 0.2, 0.4, 0.5],
    # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
    'c__class_weight': [None, 'balanced']

}

estimator = Pipeline([
    ('i', SimpleImputer()), # Replace nan's with the median value between previous and next observation
    ('s', RobustScaler()), # Scale data in order to center it and increase robustness against noise and outliers
    ('c', SVC()),
])