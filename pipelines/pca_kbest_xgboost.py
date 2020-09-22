from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
from xgboost import XGBClassifier
TARGET='binary_bin'
PARAMETER_GRID = {
    'f__pca__n_components':[2, 3, 4, 8],
    'f__kbest__k':[10, 20],
    'c__n_estimators': [100, 500],
    'c__objective': ['binary:logistic'],
    'c__num_parallel_tree': [1, 2, 3],
    'c__max_depth': [1, 2, 3],
    'c__reg_alpha': [0],
    'c__reg_lambda': [1],
    'c__learning_rate': [0.01, 0.005]
}

estimator = Pipeline([
    ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
    ('s', RobustScaler()),  # Scale data in order to center it and increase robustness against noise and outliers
    ('f', FeatureUnion([
        ('pca', PCA()),
        ('kbest', SelectKBest())
    ])),
    ('c', XGBClassifier()),
])