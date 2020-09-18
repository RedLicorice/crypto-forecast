from imblearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer


TARGET='binary_bin'

PARAMETER_GRID = {
    'n_estimators': [10], # Number of estimators to use in ensemble
    'max_samples': [0.5, 0.8, 1.0], # Number of samples per estimator in the ensemble
    'max_features': [0.5, 1.0], # Number of features per estimator in the ensemble
    'base_estimator__c__penalty': ['l2'], # Specifies the norm used in the penalization. The ‘l2’ penalty is the standard used in SVC.
    'base_estimator__c__loss': ['squared_hinge'], #  ‘hinge’ is the standard SVM loss (used e.g. by the SVC class) while ‘squared_hinge’ is the square of the hinge loss.
    'base_estimator__c__C': [0.1, 0.5, 1, 1.5, 2], # Regularization parameter. The strength of the regularization is inversely proportional to C. >0
    'base_estimator__c__class_weight':[None, 'balanced']
}

pipeline = Pipeline([
    ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
    ('s', RobustScaler()),
    ('c', LinearSVC()),
])

estimator = BaggingClassifier(base_estimator=pipeline)