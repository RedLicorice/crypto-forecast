import logging
from build_datasets_old import load_dataset
from lib.log import logger
from lib.dataset import target_price_variation, target_discrete_price_variation, target_binned_price_variation, discretize_ta_features
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
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

SVC_PARAM_GRID = {
    'n_estimators': [10], # Number of estimators to use in ensemble
    'max_samples': [0.5, 0.8, 1.0], # Number of samples per estimator in the ensemble
    'max_features': [0.5, 1.0], # Number of features per estimator in the ensemble
    'base_estimator__c__C': [0.1, 0.5, 1, 1.5, 2],
    'base_estimator__c__kernel': ['linear', 'poly'],
    'base_estimator__c__gamma': ['auto'],
    'c__base_estimator__degree': [2, 3, 4, 5, 10], # only makes sense for poly
    'base_estimator__c__coef0': [0.0, 0.1, 0.2, 0.4, 0.5],
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
    index = load_dataset('all_merged', return_index=True, index_name='index_improved')
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

            # Derive target classes from closing price
            target_pct = target_price_variation(features['close'])
            target = target_binned_price_variation(target_pct, n_bins=2)
            # target = target_discrete_price_variation(target_pct)

            # Use selected features
            preselected = hyperparameters[_sym]['features']
            features = features[preselected]

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

            # Build pipeline to be used as estimator in bagging classifier
            #  so that each subset of the data is transformed independently
            #  to avoid contamination between folds.
            pipeline = Pipeline([
                ('i', SimpleImputer()),  # Replace nan's with the median value between previous and next observation
                ('s', RobustScaler()),
                ('c', SVC()),
            ])
            # The base estimator to be tuned is the Bagging ensemble
            #  composed of multiple instances of our pipeline
            ensemble = BaggingClassifier(base_estimator=pipeline)

            # Perform hyperparameter tuning of the ensemble with 5-fold cross validation
            logger.info("Start Grid search")
            CV_rfc = GridSearchCV(
                estimator=ensemble,
                param_grid=SVC_PARAM_GRID,
                cv=5,
                n_jobs=4,
                scoring='neg_mean_squared_error',
                verbose=1
            )
            CV_rfc.fit(X_train, y_train)
            logger.info("End Grid search")

            clf1 = pipeline.fit(X_train, y_train)
            # Take the fitted ensemble with tuned hyperparameters
            clf = CV_rfc.best_estimator_
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
            ##
            clf = clf1
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
        filename='../build_bagging_model.log',
        filemode='w',
        root_level=logging.DEBUG,
        log_level=logging.DEBUG,
        logger='build_bagging_model'
    )
    main()
