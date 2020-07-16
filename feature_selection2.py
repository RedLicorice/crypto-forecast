import logging
from lib.log import logger
import pandas as pd
from multiprocessing import freeze_support
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV
from genetic_selection import GeneticSelectionCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import json

def randomforest(X, y, columns):
    forest = RandomForestClassifier(n_estimators=250, random_state=0)
    forest.fit(X,y)
    importances = {columns[i]:v for i, v in enumerate(forest.feature_importances_)}
    labeled = {str(k): v for k, v in sorted(importances.items(), key=lambda item: -item[1])}
    return {
        'feature_importances': labeled,
        'rank': {l:i+1 for i, l in enumerate(labeled.keys())}
    }

def randomforest_rfecv(X, y, columns):
    estimator = RandomForestClassifier(n_estimators=250, random_state=0)
    selector = RFECV(estimator, step=1, cv=5, verbose=0)
    selector = selector.fit(X, y)
    # selector ranking to column:rank pairs
    rank = {columns[i]: s for i, s in enumerate(selector.ranking_)}
    # Feature importances
    importances = {columns[i]: v for i, v in enumerate(selector.estimator_.feature_importances_)}
    labeled = {str(k): v for k, v in sorted(importances.items(), key=lambda item: -item[1])}

    return {
        # sort rank by values
        'rank': {str(k): int(v) for k, v in sorted(rank.items(), key=lambda item: item[1])},
        # pick selected features names
        'support': [columns[i] for i, s in enumerate(selector.support_) if s],
        'feature_importances': labeled
    }

def randomforest_genetic(X, y, columns):
    estimator = RandomForestClassifier(n_estimators=250, random_state=0)
    selector = GeneticSelectionCV(estimator,
                                      cv=5,
                                      verbose=0,
                                      scoring="accuracy",
                                      max_features=min(X.shape[1], 30),
                                      n_population=50,
                                      crossover_proba=0.5,
                                      mutation_proba=0.2,
                                      n_generations=80,
                                      crossover_independent_proba=0.5,
                                      mutation_independent_proba=0.05,
                                      tournament_size=5,
                                      n_gen_no_change=10,
                                      caching=True,
                                      n_jobs=-1)
    selector = selector.fit(X, y)
    support_names = [columns[i] for i, s in enumerate(selector.support_) if s]
    importances = {columns[i]: v for i, v in enumerate(selector.estimator_.feature_importances_)}
    labeled = {str(k): v for k, v in sorted(importances.items(), key=lambda item: -item[1])}
    return {
        # pick selected features names
        'support': support_names,
        # pick feature coefficients
        #'coef': {support_names[i]: c for i, c in enumerate(selector.estimator_.coef_)},
        'feature_importances': labeled
    }

def main(dataset):
    indexFile = 'data/datasets/{}/index.json'.format(dataset)
    resultFile = 'data/datasets/{}/feature_selection2.json'.format(dataset)
    with open(indexFile) as f:
        index = json.load(f)

    result = {}
    for _sym, files in index.items():
        df = pd.read_csv(files['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        X = df[df.columns.difference(['target', 'target_label','target_pct'])]
        y = df[['target']]['target'].values
        run = {}
        # Split train and test set in an expanding window fashion (decrease test set)
        for testSize in [0.3]:
            logger.info("Processing {} testSize: {}".format(_sym, testSize))
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=testSize)
            run["test_size={}".format(testSize)] = {
                'genetic': randomforest_genetic(X_train, y_train, X.columns),
                'rfecv': randomforest_rfecv(X_train, y_train, X.columns),
                'randomforest': randomforest(X_train, y_train, X.columns)
            }
        result[_sym] = run

    with open(resultFile, 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == '__main__':
    freeze_support()
    logger.setup(
        filename='../feature_selection2.log',
        filemode='w',
        root_level=logging.DEBUG,
        log_level=logging.DEBUG,
        logger='correlation'
    )
    main('ohlcv_coinmetrics')
    main('ohlcv_social')
    main('resampled_ohlcv_ta')
