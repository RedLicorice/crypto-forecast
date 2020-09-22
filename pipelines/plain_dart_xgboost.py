from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


TARGET='binary_bin'

PARAMETER_GRID = {
    'i__strategy': ['mean'],  # 'median', 'most_frequent', 'constant'
    'c__booster': ['dart'], # Use DART booster, adds dropout techniques from the deep neural net community to boosted trees, and reported better results in some situations.
    'c__sample_type': ['uniform', 'weighted'], # 'uniform': dropped trees selected uniformly | 'weighted': dropped trees selected in proportion to weight
    'c__normalize_type': ['tree'], # Tree: new trees have same weight of dropped trees | 'forest': new trees have same weight of sum of dropped trees
    'c__rate_drop': [0.4], # Dropout rate [0,1]
    'c__skip_drop': [0.2], # probability to skip dropout [0,1]
    'c__n_estimators': [10, 500, 1000],
    'c__objective': ['binary:logistic'],
    'c__num_parallel_tree': [1, 2],
    'c__max_depth': [1, 2, 3],
    'c__reg_alpha': [0],
    'c__reg_lambda': [1],
    'c__learning_rate': [0.01, 0.005, 0.001]
}

estimator = Pipeline([
    ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
    ('s', StandardScaler()),  # Scale data in order to center it and increase robustness against noise and outliers
    ('c', XGBClassifier()),
])