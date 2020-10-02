import logging
from build_datasets_old import load_dataset
from lib.log import logger
from lib.dataset import target_price_variation, target_discrete_price_variation, target_binned_price_variation, discretize_ta_features
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve
from collections import Counter
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, RobustScaler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import os
import pickle
import json

RANDOMFOREST_PARAM_GRID = {
    'c__n_estimators': [10, 100, 200, 500],
    'i__strategy': ['mean'],  # 'median', 'most_frequent', 'constant'
    'c__criterion': ['gini'],  # , 'entropy'],
    'c__max_depth': [4, 8],
    'c__min_samples_split': [2],
    'c__min_samples_leaf': [1, 0.05, 0.1, 0.2],
    'c__max_features': ['auto', 'log2', 0.1, 0.5],  # 'sqrt',
    'c__class_weight': [None, 'balanced', 'balanced_subsample'],
}

def plot_class_distribution(dataset, _sym, y):
    counter = Counter(y)
    print('Dataset: {} Symbol: {}'.format(dataset, _sym))
    for k, v in counter.items():
        per = v / len(y) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    # plot the distribution
    #plt.title(_sym)
    #plt.bar(counter.keys(), counter.values())
    #plt.show()

def main():
    index = load_dataset('all_merged', return_index=True)
    resultFile = './data/datasets/all_merged/estimators/randomforest_sfm_hyperparameters.json'
    hyperparameters = {}
    if not os.path.exists(resultFile):
        logger.error('no hyperparameters!')
    with open(resultFile, 'r') as f:
        hyperparameters = json.load(f)
    for _sym, data in index.items():
        if _sym not in hyperparameters or not os.path.exists(hyperparameters[_sym]['estimator']):
            logger.error('{} does not exist.'.format(_sym))
        else:
            features = pd.read_csv(data['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
            # Replace nan with infinity so that it can later be imputed to a finite value
            features = features.replace([np.inf, -np.inf], np.nan)
            #features = features[hyperparameters['feature_importances']]

            # Derive target classes from closing price
            target_pct = target_price_variation(features['close'])
            target = target_binned_price_variation(target_pct, n_bins=2)
            # target = target_discrete_price_variation(target_pct)

            # Split data in train and blind test set with 70:30 ratio,
            #  most ML models don't take sequentiality into account, but our pipeline
            #  uses a SimpleImputer with mean strategy, so it's best not to shuffle the data.
            X_train, X_test, y_train, y_test = train_test_split(features.values, target.values, shuffle=False,
                                                                test_size=0.3)
            # Summarize distribution
            print("Training set: # Features {}, # Samples {}".format(X_train.shape[1], X_train.shape[0]))
            plot_class_distribution("Training set", _sym, y_train)
            print("Test set: # Features {}, # Samples {}".format(X_test.shape[1], X_test.shape[0]))
            plot_class_distribution("Test set", _sym, y_test)
            if not np.isfinite(X_train).all():
                logger.warning("Training x is not finite!")
            if not np.isfinite(y_train).all():
                logger.warning("Training y is not finite!")
            if not np.isfinite(X_test).all():
                logger.warning("Test x is not finite!")
            if not np.isfinite(y_test).all():
                logger.warning("Test y is not finite!")

            # Take the fitted ensemble with tuned hyperparameters
            clf = None
            with open(hyperparameters[_sym]['estimator'], 'rb') as f:
                clf = pickle.load(f)


            # Test ensemble's performance on training and test sets
            logger.info("Classification report on train set")
            predictions1 = clf.predict(X_train)
            train_report = classification_report(y_train, predictions1, output_dict=True)
            print(classification_report(y_train, predictions1))
            logger.info("Classification report on test set")
            predictions2 = clf.predict(X_test)
            test_report = classification_report(y_test, predictions2, output_dict=True)
            print(classification_report(y_test, predictions2))
            stats = {
                'score': accuracy_score(y_train, predictions1),
                'mse': mean_squared_error(y_train, predictions1),
                'test_score': accuracy_score(y_test, predictions2),
                'test_mse': mean_squared_error(y_test, predictions2),
                'train_report': train_report,
                'test_report': test_report,
            }
            print(stats)
            print("--- end ---")

if __name__ == '__main__':
    logger.setup(
        filename='../test_pickled_model.log',
        filemode='w',
        root_level=logging.DEBUG,
        log_level=logging.DEBUG,
        logger='test_pickled_model'
    )
    main()
