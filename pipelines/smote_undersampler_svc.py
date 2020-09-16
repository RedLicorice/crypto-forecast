from imblearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

PARAMETER_GRID = {
    'c__C': [0.5, 1, 1.5, 2],
    'c__kernel': ['poly'], #'linear', 'rbf'
    'c__gamma': ['scale', 'auto'],
    'c__degree': [3, 4, 5, 10],
    'c__coef0': [0.0,  0.2, 0.4, 0.5, 0.8, 1.0],
}

estimator = Pipeline([
    ('i', SimpleImputer(strategy='mean')),
    ('s', MinMaxScaler()),
    ('o', SMOTE()),
    ('u', RandomUnderSampler()),
    ('c', SVC()),
])