from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from lib.selection_pipeline import SelectionPipeline
from xgboost import XGBClassifier


TARGET='binary_bin'

PARAMETER_GRID = {
    'estimator__i__strategy': ['mean'],  # 'median', 'most_frequent', 'constant'
    'estimator__c__n_estimators': [100, 500, 1000],
    'estimator__c__objective': ['binary:logistic', 'binary:hinge'],
    'estimator__c__num_parallel_tree': [1],
    'estimator__c__max_depth': [1, 2],
    'estimator__c__subsample': [1, 0.8],
    'estimator__c__reg_alpha': [0], # L1 regularization term on weights. Increasing this value will make model more conservative.
    'estimator__c__reg_lambda': [1], # L2 regularization term on weights. Increasing this value will make model more conservative.
    'estimator__c__eta': [0.3], # Step size shrinkage used in update to prevents overfitting. [0,1]
    'estimator__c__gamma': [0], # Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.
    'estimator__c__learning_rate': [0.01, 0.005]
}

pipeline = SelectionPipeline([
    ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
    ('s', StandardScaler()),  # Scale data in order to center it and increase robustness against noise and outliers
    ('c', XGBClassifier(zero_division=1)),
])

estimator = RFE(estimator=pipeline)