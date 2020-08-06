# Perform feature selection with randomforest + rfecv but use params
# from grid search
import logging
from lib.log import logger
import pandas as pd
from multiprocessing import freeze_support
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV
from genetic_selection import GeneticSelectionCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import numpy as np
import json
GRIDSEARCH_CLASSIFIER_PARAMS = {
    'bootstrap': True, 'class_weight': 'balanced',
    #'class_weight': None,
    'criterion': 'gini',
    'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0, 'min_impurity_split': None,
    'min_samples_leaf': 1, 'min_samples_split': 2,
    'min_weight_fraction_leaf': 0.0, 'n_estimators': 10, 'n_jobs': 1,
    'oob_score': False, 'random_state': 0, 'verbose': 0, 'warm_start': False
}
CLASSIFIER_PARAM_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced']
}
CLASSIFIER_PARAMS = {
    'n_estimators':250,
    'random_state':0,
    'oob_score':True,# Default is False
    'min_samples_leaf':1, # Minimum number of samples required to be at a leaf node. Default is 1, if float is percent of samples
    'max_depth': 8, # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    'min_samples_split': 0.05,# Minimum number of samples required to split an internal node. Default is 2, if float is percent of samples
    'max_features':'log2'# Number of features to consider when looking for the best split. Default is 'auto'
}

def randomforest_gridsearch(X, y, X_test, y_test, columns):
    logger.info("Start RandomForest Grid search")
    CV_rfc = GridSearchCV(
        estimator=RandomForestClassifier(**GRIDSEARCH_CLASSIFIER_PARAMS),
        param_grid=CLASSIFIER_PARAM_GRID,
        cv=5
    )
    CV_rfc.fit(X, y)
    forest = CV_rfc.best_estimator_
    logger.info("End RandomForest Grid search")

    importances = {columns[i]:v for i, v in enumerate(forest.feature_importances_)}
    labeled = {str(k): v for k, v in sorted(importances.items(), key=lambda item: -item[1])}

    return {
        'feature_importances': labeled,
        'rank': {l:i+1 for i, l in enumerate(labeled.keys())},
        'score': forest.score(X,y),
        'test_score': forest.score(X_test, y_test),
        'cv_best_score': CV_rfc.best_score_,
        #'cv_results': CV_rfc.cv_results_,
        'cv_bestparams': CV_rfc.best_params_
    }

def randomforest_rfecv(X, y, X_test, y_test, columns):
    logger.info("Start RandomForest + RFECV")
    estimator = RandomForestClassifier(**CLASSIFIER_PARAMS)
    selector = RFECV(estimator, step=1, cv=5, verbose=0)
    selector = selector.fit(X, y)
    logger.info("End RandomForest + RFECV")
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
        'feature_importances': labeled,
        'score': selector.score(X,y),
        'test_score': selector.score(X_test, y_test)
    }

def randomforest_genetic(X, y, X_test, y_test, columns):
    logger.info("Start RandomForest + Genetic")
    estimator = RandomForestClassifier(**CLASSIFIER_PARAMS)
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
    logger.info("End RandomForest + Genetic")
    support_names = [columns[i] for i, s in enumerate(selector.support_) if s]
    importances = {columns[i]: v for i, v in enumerate(selector.estimator_.feature_importances_)}
    labeled = {str(k): v for k, v in sorted(importances.items(), key=lambda item: -item[1])}
    return {
        # pick selected features names
        'support': support_names,
        # pick feature coefficients
        #'coef': {support_names[i]: c for i, c in enumerate(selector.estimator_.coef_)},
        'feature_importances': labeled,
        'score': selector.score(X,y),
        'test_score': selector.score(X_test, y_test)
    }

def main(dataset):
    indexFile = 'data/datasets/{}/index.json'.format(dataset)
    resultFile = 'data/datasets/{}/feature_selection5.json'.format(dataset)
    with open(indexFile) as f:
        index = json.load(f)

    result = {}
    for _sym, files in index.items():
        df = pd.read_csv(files['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        features = [c for c in df.columns.difference(['target', 'target_label','target_pct']) if not c.endswith('_p1') and not c.endswith('_d1')]
        pct_features = features + [c for c in df.columns if c.endswith("_p1")]
        diff_features = features + [c for c in df.columns if c.endswith("_d1")]
        _X = df[df.columns.difference(['target', 'target_label','target_pct'])]
        y = df[['target']]['target'].values
        testSize = 0.3
        run = []
        # Split train and test set in an expanding window fashion (decrease test set)
        for X, set in [(_X[pct_features], "pct")]:#, (_X[diff_features], "diff")]:
            logger.info("Processing {}.{} set [{}] testSize: {} ".format(dataset, _sym, set, testSize))
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=testSize)
            gsr = randomforest_gridsearch(X_train, y_train, X_test, y_test, X.columns)
            CLASSIFIER_PARAMS = gsr['cv_bestparams']
            run.append({
                'test_size': testSize,
                'feature_set': set,
                'randomforest': gsr,
                'genetic': randomforest_genetic(X_train, y_train, X_test, y_test, X.columns),
                'rfecv': randomforest_rfecv(X_train, y_train, X_test, y_test, X.columns),
            })
            result[_sym] = run
        break  ### /!\ Only does first item!

    with open(resultFile, 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == '__main__':
    freeze_support()
    logger.setup(
        filename='../feature_selection4.log',
        filemode='w',
        root_level=logging.DEBUG,
        log_level=logging.DEBUG,
        logger='feature_selection'
    )
    main('ohlcv_coinmetrics')
    # main('ohlcv_social')
    # main('resampled_ohlcv_ta')
