import logging
from lib.log import logger
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from collections import Counter
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot as plt
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
RANDOMFOREST_PARAM_GRID = {
    'c__n_estimators': [200, 500, 1000],
    'c__max_features': ['auto', 'sqrt', 'log2'],
    'c__max_depth': [4, 5, 6, 7, 8],
    'c__criterion': ['gini', 'entropy'],
    'c__class_weight': [None, 'balanced', 'balanced_subsample']
}
SVC_PARAM_GRID = {
    'c__C': [0.5, 1, 1.5, 2],
    'c__kernel': ['poly'], #'linear', 'rbf'
    'c__gamma': ['scale', 'auto'],
    'c__degree': [3, 4, 5, 10],
    'c__coef0': [0.0,  0.2, 0.4, 0.5, 0.8, 1.0],
}

def plot_class_distribution(dataset, _sym, y):
    counter = Counter(y)
    print('Dataset: {} Symbol: {}'.format(dataset, _sym))
    for k, v in counter.items():
        per = v / len(y) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    # plot the distribution
    plt.title(_sym)
    plt.bar(counter.keys(), counter.values())

def main(dataset):
    indexFile = 'data/datasets/{}/index.json'.format(dataset)
    #resultFile = 'data/datasets/{}/feature_selection5.json'.format(dataset)
    with open(indexFile) as f:
        index = json.load(f)

    #result = {}
    for _sym, files in index.items():
        df = pd.read_csv(files['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        df = df.replace([np.inf, -np.inf], np.nan)#.dropna()
        features = [c for c in df.columns.difference(['target', 'target_label','target_pct']) if not c.endswith('_d1')]
        X = df[features]
        y = df[['target']]['target'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

        # summarize distribution
        print("Training set: # Features {}, # Samples {}".format(X_train.shape[1], X_train.shape[0]))
        plot_class_distribution(dataset, _sym, y_train)
        pipeline = Pipeline([
            ('i', SimpleImputer(strategy='mean')),
            ('s', MinMaxScaler()),
            ('o', SMOTE()),
            ('u', RandomUnderSampler()),
            ('c', SVC()),
        ])
        #X_enc, y_enc = pipeline.fit_resample(X_train, y_enc)
        #plot_class_distribution(dataset, _sym, y_enc)
        logger.info("Start Grid search")
        CV_rfc = GridSearchCV(
            estimator=pipeline, # RandomForestClassifier(**GRIDSEARCH_CLASSIFIER_PARAMS),
            # param_grid=RANDOMFOREST_PARAM_GRID,
            param_grid=SVC_PARAM_GRID,
            cv=5,
            n_jobs=4,
            scoring='neg_mean_squared_error'
        )
        CV_rfc.fit(X_train, y_train)
        # CV_rfc.fit(X_enc, y_enc)
        clf = CV_rfc.best_estimator_#.named_steps.c
        logger.info("End Grid search")

        #importances = {X.columns[i]: v for i, v in enumerate(forest.feature_importances_)}
        #labeled = {str(k): v for k, v in sorted(importances.items(), key=lambda item: -item[1])}

        print({
            #'feature_importances': labeled,
            #'rank': {l: i + 1 for i, l in enumerate(labeled.keys())},
            'score': clf.score(X_train, y_train),
            'test_score': clf.score(X_test, y_test),
            'cv_best_score': CV_rfc.best_score_,
            # 'cv_results': CV_rfc.cv_results_,
            'cv_bestparams': CV_rfc.best_params_
        })
        print("--- end ---")



    #with open(resultFile, 'w') as f:
    #    json.dump(result, f, indent=4)


if __name__ == '__main__':
    logger.setup(
        filename='../dataset_info.log',
        filemode='w',
        root_level=logging.DEBUG,
        log_level=logging.DEBUG,
        logger='dataset_info'
    )
    #main('ohlcv_coinmetrics')
    # main('ohlcv_social')
    main('resampled_ohlcv_ta')
