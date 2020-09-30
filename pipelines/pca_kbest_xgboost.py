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
    'c__n_estimators': [10, 100, 500],
    'c__objective': ['binary:logistic'],
    'c__eval_metric': ['aucpr'], # Evaluation metrics for validation data during tree building
    'c__subsample': [1], # Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees.
    'c__colsample_bytree': [1, 0.8, 0.5], # Subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
    'c__colsample_bylevel': [1, 0.8, 0.5], # Subsample ratio of columns for each level. Subsampling occurs once for every new depth level reached in a tree. Columns are subsampled from the set of columns chosen for the current tree.
    'c__colsample_bynode': [1, 0.8, 0.4],# Subsample ratio of columns for each node (split). Subsampling occurs once every time a new split is evaluated. Columns are subsampled from the set of columns chosen for the current level.
    'c__num_parallel_tree': [1, 2], # Number of parallel trees constructed during each iteration. This option is used to support boosted random forest.
    'c__max_depth': [1, 2, 3], # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
    'c__reg_alpha': [0], # L1 regularization term on weights. Increasing this value will make model more conservative.
    'c__reg_lambda': [1], # L2 regularization term on weights. Increasing this value will make model more conservative.
    'c__learning_rate': [0.001, 0.005], # 0.3 Step size shrinkage used in update to prevents overfitting. Shrinks the feature weights to make the boosting process more conservative.
    'c__scale_pos_weight': [1] # should be negative_samples_count / positive_samples_count
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