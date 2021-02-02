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


# Perform Cross-Validated grid search with a given estimator over a given
# parameter grid, return search results
def grid_search(est, parameters, X_train, y_train, **kwargs):
    cv = kwargs.get('cv', 5)
    expanding_window = kwargs.get('expanding_window', False)
    n_jobs = kwargs.get('n_jobs', 'auto')
    scoring = kwargs.get('scoring', 'accuracy')
    # Parse arguments
    if n_jobs == 'auto':
        n_jobs = os.cpu_count()
    elif n_jobs.isnumeric():
        n_jobs = int(n_jobs)
    if expanding_window:
        cv = TimeSeriesSplit(n_splits=expanding_window)
    if ',' in scoring:
        scoring = scoring.split(',')
    # if scoring is precision, make scorer manually to suppress zero_division warnings in case of heavy bias
    if scoring == 'precision':
        scoring = make_scorer(precision_score, zero_division=1, average='micro')

    gscv = GridSearchCV(
        estimator=est,
        param_grid=parameters,
        cv=cv,
        n_jobs=n_jobs,
        scoring=scoring,
        verbose=1
    )
    gscv.fit(X_train, y_train)
    return gscv

def test_model(est, parameters, W,  X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array, **kwargs):
    result = np.nan
    if X_train.shape[0] <= W:
        logger.warning("Too few training datapoints for window {}".format(W))
        return result
    features = np.concatenate((X_train[:-W], X_test))
    target = np.concatenate((y_train[:-W], y_test))

    predictions = []
    labels = []

    # Go in reverse
    window = 1
    for i in range(features.shape[0], 0, -1):
        if i < (W + 1):
            break
        train_start = i -W -1
        train_end = i -1
        test_start = i-1
        test_end = i
        # print('[Window {}]\tTrain: B={} E={}\tTest: B={} E={}'.format(window, train_start, train_end, test_start, test_end))
        _X_train = features[train_start:train_end]
        _y_train = target[train_start:train_end]
        _X_test = features[test_start:test_end]
        _y_test = target[test_start:test_end]

        _est = est.set_params(**parameters)
        _est = _est.fit(_X_train, _y_train)
        pred = _est.predict(_X_test)
        
        predictions.append(pred[0])
        labels.append(_y_test[0])
        window += 1
        # print('\t Expect: {} Predict: {}'.format(_y_test[0], pred[0]))

    labels_arr = np.flip(np.array(labels), axis=0)
    predictions_arr = np.flip(np.array(predictions), axis=0)
    print('======== Final score! =======')
    classification_report(labels_arr, predictions_arr)
    return (labels_arr, predictions_arr)

def build_model(dataset, pipeline, experiment, param_grid=None, cv=5, scoring='accuracy', n_jobs='auto', test_size=0.3, use_target=None, expanding_window=False):
    # Define log file path for this run
    log_file = './results/{}_{}_{}/model_build.log'.format(dataset, pipeline, experiment)
    os.makedirs('./results/{}_{}_{}'.format(dataset, pipeline, experiment), exist_ok=True)
    # Setup logging
    logger.setup(
        filename=log_file,
        filemode='w',
        root_level=logging.DEBUG,
        log_level=logging.DEBUG,
        logger='build_model'
    )

    # Load the dataset index
    dataset_index = load_dataset(dataset, return_index=True)
    # Dynamically import the pipeline we want to use for building the model
    p = importlib.import_module('pipelines.' + pipeline)


    # Parameter grid argument:
    # - If None, use the pipeline-defined grid
    # - If string, parse it as JSON
    # - If dict, use it as-is (do nothing)
    if param_grid == None:
        param_grid = p.PARAMETER_GRID
    elif type(param_grid) is 'str':
        with open(param_grid, 'r') as f:
            param_grid = json.load(f)
    # Target argument
    # Determines the target feature name (the system supports different classification targets)
    # If not supplied, use the pipeline-defined target.
    current_target = p.TARGET if not use_target else use_target


    logger.info('Start processing: {} using {} on {} with target {}'.format(experiment, pipeline, dataset, current_target))
    reports = ReportCollection(dataset, pipeline, experiment)
    for _sym, data in dataset_index.items():
        try:
            logger.info('Start processing: {}'.format(_sym))
            # ToDo: use lib.dataset.features.load_symbol instead of manually reading csv's
            features = pd.read_csv(data['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
            targets = pd.read_csv(data['target_csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)

            # Drop columns whose values are all NaN, as well as rows with ANY nan value, then
            # replace infinity values with nan so that they can later be imputed to a finite value
            # in the pipeline's "Inputing" stage
            features = features.dropna(axis='columns', how='all').replace([np.inf, -np.inf], np.nan)
            target = targets.loc[features.index][current_target]

            # Split available data in train and test set.
            # Perform grid search with cross-validation on the training set,
            # Then ToDo: instantiate a MLStrategy and test the model on the test set, in sliding window fashion
            X_train, X_test, y_train, y_test = train_test_split(features.values, target.values, shuffle=False, test_size=test_size)

            # Log before and after grid search to track execution time
            # Grid search logic moved to its own method for cleanliness
            logger.info("Start Grid search")
            gscv = grid_search(p.estimator,
                             param_grid,
                             X_train,
                             y_train,
                             cv=cv,
                             n_jobs=n_jobs,
                             expanding_window=expanding_window,
                             scoring=scoring
                )
            logger.info("End Grid search")
            labels, predictions = test_model(p.estimator, gscv.best_params_, 30, X_train, y_train, X_test, y_test)
            report = classification_report(labels, predictions, output_dict=True)

            # Create a Report instance from the grid search results, and add it to this experiment's collection
            _report = Report(_sym, current_target, cv)
            _report.set_close(targets.loc[features.index].close)
            _report.set_dataset_columns(features.columns)
            _report.set_train_dataset(X_train, y_train)
            _report.set_test_dataset(X_test, y_test)
            _report.set_model(p.estimator)
            _report.set_params(gscv.best_params_)
            _report.set_cv(gscv.best_estimator_, gscv.best_score_, gscv.cv_results_)
            reports.add_report(_report)
            reports.save()

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
