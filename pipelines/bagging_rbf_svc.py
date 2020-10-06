from imblearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


TARGET='binary_bin'

PARAMETER_GRID = {
    'n_estimators': [10], # Number of estimators to use in ensemble
    'max_samples': [0.5, 0.8, 1.0], # Number of samples per estimator in the ensemble
    'max_features': [0.5, 1.0], # Number of features per estimator in the ensemble
    'base_estimator__c__C': [1, 1.5, 2], # Regularization parameter. The strength of the regularization is inversely proportional to C. >0
    'base_estimator__c__kernel': ['rbf'],
    'base_estimator__c__gamma': ['scale', 'auto'], # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. (default = 'scale')
    'base_estimator__c__class_weight':[None, 'balanced']
}

pipeline = Pipeline([
    ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
    ('s', MinMaxScaler()),
    ('c', SVC()),
])

estimator = BaggingClassifier(base_estimator=pipeline)