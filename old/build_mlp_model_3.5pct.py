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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif, RFECV
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve
from collections import Counter
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, RobustScaler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import os
import pickle
import json, math

PARAM_GRID = {
    # 'c__n_estimators': [10, 100, 200, 500],
    # 'c__criterion': ['gini'],  # , 'entropy'],
    # 'c__max_depth': [4, 8],
    # 'c__min_samples_split': [2],
    # 'c__min_samples_leaf': [1, 0.05, 0.1, 0.2],
    # 'c__max_features': ['auto', 'log2', 0.1, 0.5],  # 'sqrt',
    # 'c__class_weight': [None, 'balanced', 'balanced_subsample']
    'c__hidden_layer_sizes':[(2,), (4,), (2,4), (4,8)],
    'c__solver':['adam'],
    'c__activation':['logistic','tanh','relu'],
    'c__alpha':[0.0001, 0.001, 0.01],
    'c__learning_rate':['constant','adaptive'],
    'c__random_state':[0],
    'c__max_iter':[2000]
}

def test_gains(close, y_pred, initial_balance=100, position_size=0.1):
    position_amount = initial_balance*position_size
    balance=initial_balance
    coins = 0
    last_price = None
    for price, y in zip(close, y_pred):
        if not price or np.isnan(price):
            continue
        if y not in [0, 1]:
            continue
        if not y: # Sell if y == 0
            amount = position_amount/price
            if coins < amount:
                amount = coins
            coins -= amount
            balance += amount*price
        else:# Buy if y == 1
            amount = position_amount/price
            if balance < position_amount:
                amount = balance/price
            balance -= amount*price
            coins += amount
        last_price = price
    if coins and last_price:
        balance += coins*last_price
        coins = 0
    return balance



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

def get_symbol_features(index, sym):
    data = index[sym]
    features = pd.read_csv(data['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
    # Replace nan with infinity so that it can later be imputed to a finite value
    features = features.replace([np.inf, -np.inf], np.nan)

    # Derive target classes from closing price
    target_pct = target_price_variation(features['close'])
    target = target_binned_price_variation(target_pct, n_bins=2)
    return features, target

def main():
    index = load_dataset('all_merged', return_index=True)
    for _sym, data in index.items():
        features, target = get_symbol_features(index, _sym)

        features_p = features[data['features']['ohlcv']].pct_change().replace([np.inf, -np.inf], np.nan)
        features_p.columns = [c + '_p1' for c in features_p.columns]
        features_1 = features_p.shift(1)
        features_1.columns = [c+'_lag1' for c in features_1.columns]
        features_2 = features_p.shift(2)
        features_2.columns = [c+'_lag2' for c in features_2.columns]
        ta = features[data['features']['ta']]

        features = pd.concat([features['close'], ta, features_p, features_1, features_2], axis=1)[30:]
        target = target[30:]
        # Split data in train and blind test set with 70:30 ratio,
        #  most ML models don't take sequentiality into account, but our pipeline
        #  uses a SimpleImputer with mean strategy, so it's best not to shuffle the data.
        X_train, X_test, y_train, y_test = train_test_split(features.values, target.values, shuffle=False,
                                                            test_size=0.3)
        imp = SimpleImputer()
        values = imp.fit_transform(X_train)
        #sel = SelectKBest(score_func=f_classif, k=min(10, X_train.shape[1]))
        feature_count = int(0.3*X_train.shape[1])
        sel = RFECV(
            estimator=RandomForestClassifier(),
            cv=5,
            verbose=1,
            n_jobs=4,
            min_features_to_select=feature_count,
            scoring='neg_mean_squared_error'
        )
        sel.fit(values, y_train)
        bestfeatures = [c for c, f in zip(features.columns, sel.get_support()) if f]
        if not 'close' in bestfeatures:
            bestfeatures += ['close']
        print("Using features:\n{}".format(bestfeatures))

        train_features = pd.DataFrame(X_train, columns=features.columns)[bestfeatures]
        test_features = pd.DataFrame(X_test, columns=features.columns)[bestfeatures]
        X_train = train_features.values
        X_test = test_features.values
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

        # Build pipeline to be used as estimator in grid search
        #  so that each subset of the data is transformed independently
        #  to avoid contamination between folds.
        pipeline = Pipeline([
            ('i', IterativeImputer()),  # Replace nan's with the median value between previous and next observation
            ('s', RobustScaler()),
            ('c', MLPClassifier()),
        ])

        # Perform hyperparameter tuning of the ensemble with 5-fold cross validation
        logger.info("Start Grid search")
        CV_rfc = GridSearchCV(
            estimator=pipeline,
            param_grid=PARAM_GRID,
            cv=5,
            n_jobs=4,
            scoring='neg_mean_squared_error',
            verbose=1
        )
        CV_rfc.fit(X_train, y_train)
        logger.info("End Grid search")

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
        print(CV_rfc.best_params_)
        num_samples = min(y_train.shape[0], y_test.shape[0], 30)
        print("Gains calculated on {} samples only!".format(num_samples))
        print("Train Accuracy: {}\nTrain MSE: {}\nGains on train preds: 100 -> {}".format(
            accuracy_score(y_train, predictions1),
            mean_squared_error(y_train, predictions1),
            test_gains(train_features['close'][0:num_samples], predictions1[0:num_samples], initial_balance=100, position_size=0.1)
        ))
        print("Test Accuracy: {}\nTest MSE: {}\nGains on test preds: 100 -> {}".format(
            accuracy_score(y_test, predictions2),
            mean_squared_error(y_test, predictions2),
            test_gains(test_features['close'][0:num_samples], predictions2[0:num_samples], initial_balance=100, position_size=0.1)
        ))
        print("--- end ---")

if __name__ == '__main__':
    logger.setup(
        filename='../build_model.log',
        filemode='w',
        root_level=logging.DEBUG,
        log_level=logging.DEBUG,
        logger='build_model'
    )
    main()
