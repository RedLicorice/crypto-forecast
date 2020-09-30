from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Estimators
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

TARGET='binary_bin'

PARAMETER_GRID = {
    # KNN Parameters
    'c__knn__n_neighbors': [5],

    # SVC Parameters
    'c__svc__kernel': ['poly'],

    # XGBoost Parameters
    'c__xgboost__n_estimators': [30, 100, 500],
    'c__xgboost__objective': ['binary:logistic'],
    #'c__xgboost__eval_metric': ['aucpr'], # Evaluation metrics for validation data during tree building
    'c__xgboost__subsample': [1], # Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees.
    'c__xgboost__colsample_bytree': [0.8], # Subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
    'c__xgboost__colsample_bylevel': [0.8], # Subsample ratio of columns for each level. Subsampling occurs once for every new depth level reached in a tree. Columns are subsampled from the set of columns chosen for the current tree.
    'c__xgboost__colsample_bynode': [0.6],# Subsample ratio of columns for each node (split). Subsampling occurs once every time a new split is evaluated. Columns are subsampled from the set of columns chosen for the current level.
    'c__xgboost__num_parallel_tree': [1], # Number of parallel trees constructed during each iteration. This option is used to support boosted random forest.
    'c__xgboost__max_depth': [1, 2], # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
    'c__xgboost__reg_alpha': [0], # L1 regularization term on weights. Increasing this value will make model more conservative.
    'c__xgboost__reg_lambda': [1], # L2 regularization term on weights. Increasing this value will make model more conservative.
    'c__xgboost__learning_rate': [0.001, 0.0001], # 0.3 Step size shrinkage used in update to prevents overfitting. Shrinks the feature weights to make the boosting process more conservative.
    'c__xgboost__scale_pos_weight': [1.5] # should be negative_samples_count / positive_samples_count
}

estimators = [
    ('knn', KNeighborsClassifier()),
    ('svc', SVC(probability=True)),
    ('xgboost', XGBClassifier())
]

estimator = Pipeline([
    ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
    ('s', StandardScaler()),
    ('c', StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        passthrough=False # Do not pass training set to final estimator
    )),
])