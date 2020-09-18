from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer


TARGET='binary_bin'

PARAMETER_GRID = {
    'c__n_estimators': [100, 200, 500],
    'i__strategy': ['mean'],  # 'median', 'most_frequent', 'constant'
    'c__criterion': ['gini'],  # , 'entropy'],
    'c__max_depth': [4, 8],
    'c__min_samples_split': [2],
    'c__min_samples_leaf': [1, 0.1, 0.2],
    'c__max_features': ['auto', 'log2', 0.1, 0.2, 0.6],  # 'sqrt',
    'c__class_weight': [None],#, 'balanced', 'balanced_subsample'
}

estimator = Pipeline([
    ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
    ('s', RobustScaler()),  # Scale data in order to center it and increase robustness against noise and outliers
    ('c', RandomForestClassifier()),
])