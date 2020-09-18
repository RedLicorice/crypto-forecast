from imblearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer


TARGET='binary_bin'

PARAMETER_GRID = {
    'c__penalty': ['l2'], # Specifies the norm used in the penalization. The ‘l2’ penalty is the standard used in SVC.
    'c__loss': ['squared_hinge'], #  ‘hinge’ is the standard SVM loss (used e.g. by the SVC class) while ‘squared_hinge’ is the square of the hinge loss.
    'c__C': [0.1, 0.5, 1, 1.5, 2], # Regularization parameter. The strength of the regularization is inversely proportional to C. >0
    'c__class_weight':[None, 'balanced']
}

estimator = Pipeline([
    ('i', SimpleImputer()), # Replace nan's with the median value between previous and next observation
    ('s', RobustScaler()), # Scale data in order to center it and increase robustness against noise and outliers
    ('c', LinearSVC()),
])