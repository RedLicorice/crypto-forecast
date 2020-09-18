from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
TARGET='binary_bin'
PARAMETER_GRID = {
    'p__n_components':[2, 3, 4, 8],
    'c__n_estimators': [100, 500],
    'c__objective': ['binary:logistic'],
    'c__num_parallel_tree': [1],
    'c__max_depth': [1],
    'c__reg_alpha': [0],
    'c__reg_lambda': [1],
    'c__learning_rate': [0.01, 0.005]
}

estimator = Pipeline([
    ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
    ('s', RobustScaler()),  # Scale data in order to center it and increase robustness against noise and outliers
    ('p', PCA()),
    ('c', XGBClassifier()),
])