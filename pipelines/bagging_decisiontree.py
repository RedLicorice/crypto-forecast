from imblearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

PARAMETER_GRID = {
    'n_estimators': [10], # Number of estimators to use in ensemble
    'max_samples': [0.5, 0.8, 1.0], # Number of samples per estimator in the ensemble
    'max_features': [ 0.1, 0.2, 0.5, 1.0], # Number of features per estimator in the ensemble
    'base_estimator__i__strategy': ['mean'], #  'median', 'most_frequent', 'constant'
    'base_estimator__c__criterion': ['gini'],#, 'entropy'],
    'base_estimator__c__splitter': ['random', 'best'], # 'best',
    'base_estimator__c__max_depth': [4, 8],#None,
    'base_estimator__c__min_samples_split': [2],
    'base_estimator__c__min_samples_leaf': [1, 0.05, 0.1, 0.2],
    'base_estimator__c__min_weight_fraction_leaf': [0.0],# 0.01, 0.1],
    'base_estimator__c__max_features': ['auto',  'log2'], #'sqrt',
    'base_estimator__c__class_weight': [None, 'balanced']
}

pipeline = Pipeline([
    ('i', SimpleImputer()), # Replace nan's with the median value between previous and next observation
    ('s', RobustScaler()), # Scale data in order to center it and increase robustness against noise and outliers
    ('c', DecisionTreeClassifier()),
])

estimator = BaggingClassifier(base_estimator=pipeline)