import logging
from lib.log import logger
from lib.report import Report, ReportCollection
from lib.dataset import load_dataset, print_class_distribution, get_class_distribution
from lib.plot import plot_learning_curve
from lib.sliding_window_split import sliding_window_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, classification_report, plot_confusion_matrix
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, recall_score, precision_score
from scikitplot.metrics import plot_roc, plot_precision_recall
import numpy as np
import os
import pickle
import json
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.utils.data_container import detabularise

import argparse
from datetime import datetime
import traceback


def build_model(dataset, pipeline, experiment, current_target='class', test_size=0.3):
    models_dir = './results/{}_{}_{}/models/'.format(dataset, pipeline, experiment)
    reports_dir = './results/{}_{}_{}/reports/'.format(dataset, pipeline, experiment)
    experiment_index_file = './results/{}_{}_{}/index.json'.format(dataset, pipeline, experiment)
    log_file = './results/{}_{}_{}/model_build.log'.format(dataset, pipeline, experiment)

    scoring = make_scorer(precision_score, zero_division=1, average='micro')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    # Setup logging
    logger.setup(
        filename=log_file,
        filemode='w',
        root_level=logging.DEBUG,
        log_level=logging.DEBUG,
        logger='build_model'
    )
    index_name = 'index'
    if '.' in dataset:
        splits = dataset.split(".")
        dataset = splits[0]
        index_name = splits[1]
    # Load the dataset index
    dataset_index = load_dataset(dataset, return_index=True, index_name=index_name)
    # Dynamically import the pipeline we want to use for building the model
    logger.info('Start experiment: {} using {} on {} with target {}'.format(experiment, pipeline, dataset, current_target))
    reports = ReportCollection(dataset, pipeline, experiment)
    for _sym, data in {'BTC':dataset_index['BTC']}.items():
        try:
            logger.info('Start processing: {}'.format(_sym))
            features = pd.read_csv(data['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
            targets = pd.read_csv(data['target_csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)

            # Drop columns whose values are all NaN, as well as rows with ANY nan value, then
            # replace infinity values with nan so that they can later be imputed to a finite value
            features = features.dropna(axis='columns', how='all').dropna().replace([np.inf, -np.inf], np.nan)
            target = targets.loc[features.index][current_target]

            #X_train, X_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=test_size)

            all_size = features.shape[0]
            train_size = int(all_size * (1-test_size))
            features = detabularise(features[[c for c in features.columns if 'close' in c]])
            X_train = features.iloc[0:train_size]
            y_train = target.iloc[0:train_size]
            X_test = features.iloc[train_size:all_size]
            y_test = target.iloc[train_size:all_size]
            # Summarize distribution
            logger.info("Start Grid search")
            clf = ShapeletTransformClassifier(time_contract_in_mins=5)
            clf.fit(X_train, y_train)
            print('{} Score: {}'.format(_sym, clf.score(X_test, y_test)))
            pred = clf.predict(X_test)
            print(classification_report(y_test, pred))
            logger.info("End Grid search")

            logger.info("--- {} end ---".format(_sym))
        except Exception as e:
            logger.error("Exception while building model pipeline: {} dataset: {} symbol: {}\nException:\n{}".format(pipeline, dataset, _sym, e))
            traceback.print_exc()
    return reports

if __name__ == '__main__':
    # Set random seed to 0
    np.random.seed(0)
    build_model(
        'merged',
        'shapelet_test',
        'shapelet'
    )
