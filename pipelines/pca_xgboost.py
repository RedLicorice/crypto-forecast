from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PARAMETER_GRID = {
    'i__strategy': ['mean'],  # 'median', 'most_frequent', 'constant'
    'p__n_components':[4, 8, 12],
    'c__n_estimators': [500, 1000],
    'c__objective': ['binary:logistic'],
    'c__num_parallel_tree': [1, 2, 8],
    'c__max_depth': [1],
    'c__reg_alpha': [0],
    'c__reg_lambda': [1],
    'c__learning_rate': [0.01, 0.005, 0.001]
}

estimator = Pipeline([
    ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
    ('s', RobustScaler()),  # Scale data in order to center it and increase robustness against noise and outliers
    ('p', PCA()),
    ('c', XGBClassifier()),
])