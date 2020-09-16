from imblearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

PARAMETER_GRID = {
    'c__n_estimators': [10, 20, 50], # Number of estimators to use in ensemble
    'c__learning_rate': [1, 0.8, 0.5, 0.2],
    'c__base_estimator__criterion': ['gini'],#, 'entropy'],
    'c__base_estimator__splitter': ['random', 'best'], # 'best',
    'c__base_estimator__max_depth': [4, 8],#None,
    'c__base_estimator__min_samples_split': [2],
    'c__base_estimator__min_samples_leaf': [1, 0.05, 0.1, 0.2],
    'c__base_estimator__min_weight_fraction_leaf': [0.0],# 0.01, 0.1],
    'c__base_estimator__max_features': ['auto',  'log2'], #'sqrt',
    'c__base_estimator__class_weight': [None, 'balanced']
}

estimator = Pipeline([
    ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
    ('s', RobustScaler()),
    ('c', AdaBoostClassifier(base_estimator=DecisionTreeClassifier())),
])