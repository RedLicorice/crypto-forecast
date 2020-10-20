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
import importlib
import argparse
from datetime import datetime
import traceback


def build_model(dataset, pipeline, experiment, param_grid=None, cv=5, scoring='accuracy', n_jobs='auto', test_size=0.3, use_target=None, expanding_window=False):
    models_dir = './results/{}_{}_{}/models/'.format(dataset, pipeline, experiment)
    reports_dir = './results/{}_{}_{}/reports/'.format(dataset, pipeline, experiment)
    experiment_index_file = './results/{}_{}_{}/index.json'.format(dataset, pipeline, experiment)
    log_file = './results/{}_{}_{}/model_build.log'.format(dataset, pipeline, experiment)
    if ',' in scoring:
        scoring = scoring.split(',')
    # if scoring is precision, make scorer manually to suppress zero_division warnings in case of heavy bias
    if scoring == 'precision':
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
    p = importlib.import_module('pipelines.' + pipeline)

    if n_jobs == 'auto':
        n_jobs = os.cpu_count()
    elif n_jobs.isnumeric():
        n_jobs = int(n_jobs)
    # Load parameter grid argument
    if param_grid == None:
        param_grid = p.PARAMETER_GRID
    elif type(param_grid) is 'str':
        with open(param_grid, 'r') as f:
            param_grid = json.load(f)
    current_target = p.TARGET if not use_target else use_target
    logger.info('Start experiment: {} using {} on {} with target {}'.format(experiment, pipeline, dataset, current_target))
    reports = ReportCollection(dataset, pipeline, experiment)
    for _sym, data in dataset_index.items():
        try:
            logger.info('Start processing: {}'.format(_sym))
            features = pd.read_csv(data['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
            targets = pd.read_csv(data['target_csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)

            # Drop columns whose values are all NaN, as well as rows with ANY nan value, then
            # replace infinity values with nan so that they can later be imputed to a finite value
            features = features.dropna(axis='columns', how='all').replace([np.inf, -np.inf], np.nan)
            target = targets.loc[features.index][current_target]

            X_train, X_test, y_train, y_test = train_test_split(features.values, target.values, shuffle=False, test_size=test_size)
            # Summarize distribution
            logger.info("Start Grid search")
            if expanding_window:
                cv = TimeSeriesSplit(n_splits=expanding_window)
            #cv = sliding_window_split(X_train, 0.1)
            gscv = GridSearchCV(
                estimator=p.estimator,
                param_grid=param_grid,
                cv=cv,
                n_jobs=n_jobs,
                scoring=scoring,
                verbose=1
            )
            gscv.fit(X_train, y_train)
            logger.info("End Grid search")

            # Take the fitted ensemble with tuned hyperparameters
            clf = gscv.best_estimator_
            best_score = gscv.best_score_
            best_params = gscv.best_params_

            _report = Report(_sym, current_target, cv)
            _report.set_close(targets.loc[features.index].close)
            _report.set_dataset_columns(features.columns)
            _report.set_train_dataset(X_train, y_train)
            _report.set_test_dataset(X_test, y_test)
            _report.set_model(p.estimator)
            _report.set_params(gscv.best_params_)
            _report.set_cv(gscv.best_estimator_, gscv.best_score_, gscv.cv_results_)
            #experiment_index[_sym] = _report.save(reports_dir)
            reports.add_report(_report)
            reports.save()
            # with open(experiment_index_file, 'w') as f:
            #     json.dump(experiment_index, f, indent=4)
            logger.info("--- {} end ---".format(_sym))
        except Exception as e:
            logger.error("Exception while building model pipeline: {} dataset: {} symbol: {}\nException:\n{}".format(pipeline, dataset, _sym, e))
            traceback.print_exc()
    return reports

if __name__ == '__main__':
    # Set random seed to 0
    np.random.seed(0)
    parser = argparse.ArgumentParser(description='Build and tune model with GridSearchCV, using specified dataset and pipeline')
    parser.add_argument('-d', dest='dataset', nargs='?', default='', help="Dataset to be used for building the model, in format dataset.index") # nargs='?', default='all_merged.index_improved',
    parser.add_argument('-p', dest='pipeline', nargs='?', default='', help="Pipeline to be used for building the model, must be one of the filenames in pipelines folder")
    parser.add_argument('-e', dest='experiment', nargs='?', default='', help="Name for the current experiment")
    parser.add_argument('--cv',dest='cv', nargs='?', default=5, help="Number of folds to use for cross-validation in GridSearchCV")
    parser.add_argument('--njobs',dest='n_jobs', nargs='?', default='auto', help="Number of jobs to use for cross-validation in GridSearchCV")
    parser.add_argument('--scoring',dest='scoring', nargs='?', default='precision', help="Scorer to use for cross-validation in GridSearchCV")
    parser.add_argument('--test-size',dest='test_size', nargs='?', default=0.3, help="Portion of data to be kept for blind tests")
    parser.add_argument('--use-target',dest='use_target', nargs='?', default=None, help="Target to use when building the model (problem type)")
    parser.add_argument('--expanding-window',dest='expanding_window', nargs='?', default=False, type=int, help="Use TimeSeriesSplit instead of StratifiedKFold as cross validation provider with specified number of splits")
    parser.add_argument('--parameter-grid',dest='param_grid', nargs='?', default=None, type=int, help="Use a custom parameter grid (path to json)")
    args = parser.parse_args()

    if args.experiment == '':
        args.experiment = 'experiment_{}'.format(datetime.now().strftime("%d%m%y_%H%M%S"))
    else:
        args.experiment = '{}_{}'.format(args.experiment, datetime.now().strftime("%d%m%y_%H%M%S"))

    if args.scoring == 'mse':
        args.scoring = 'neg_mean_squared_error'

    if args.dataset == '':
        print('Missing dataset (-d argument)')
        exit(0)
    if args.pipeline == '':
        print('Missing pipeline (-p argument)')
        exit(0)
    build_model(
        args.dataset,
        args.pipeline,
        args.experiment,
        cv=args.cv,
        scoring=args.scoring,
        n_jobs=args.n_jobs,
        test_size=args.test_size,
        use_target=args.use_target,
        expanding_window=args.expanding_window,
        param_grid=args.param_grid
    )
