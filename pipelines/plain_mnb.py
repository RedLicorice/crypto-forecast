from imblearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer


TARGET='binary_bin'

PARAMETER_GRID = {
    'c__alpha':[0.2, 0.4, 0.6, 0.8, 1], # Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
}

estimator = Pipeline([
    ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
    ('s', RobustScaler()),
    ('c', MultinomialNB()),
])