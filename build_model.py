import logging
from lib.log import logger
from lib.dataset import load_dataset, print_class_distribution, get_class_distribution
from lib.plot import plot_learning_curve
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, classification_report, confusion_matrix, plot_roc_curve
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, recall_score, precision_score
import numpy as np
import os
import pickle
import json
import importlib
import argparse
from datetime import datetime


def build_model(dataset, pipeline, experiment, cv=5, scoring='accuracy', n_jobs='auto', test_size=0.3, use_target=None, expanding_window=False):
    models_dir = './results/{}_{}_{}/models/'.format(dataset, pipeline, experiment)
    reports_dir = './results/{}_{}_{}/reports/'.format(dataset, pipeline, experiment)
    experiment_index_file = './results/{}_{}_{}/index.json'.format(dataset, pipeline, experiment)
    log_file = './results/{}_{}_{}/model_build.log'.format(dataset, pipeline, experiment)
    if ',' in scoring:
        scoring = scoring.split(',')
    # if scoring is precision, make scorer manually to suppress zero_division warnings
    if scoring == 'precision':
        scoring = make_scorer(precision_score, zero_division=1)
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
    experiment_index = {}

    if n_jobs == 'auto':
        n_jobs = os.cpu_count()
    if expanding_window:
        cv = TimeSeriesSplit(n_splits=cv)
    logger.info('Start experiment: {} using {} on {}'.format(experiment, pipeline, dataset))
    for _sym, data in dataset_index.items():
        logger.info('Start processing: {}'.format(_sym))
        features = pd.read_csv(data['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        targets = pd.read_csv(data['target_csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        # Drop columns whose values are all NaN, as well as rows with ANY nan value, then
        # replace infinity values with nan so that they can later be imputed to a finite value
        features = features.dropna(axis='columns', how='all').dropna().replace([np.inf, -np.inf], np.nan)
        target = targets.loc[features.index][p.TARGET if not use_target else use_target]

        X_train, X_test, y_train, y_test = train_test_split(features.values, target.values, shuffle=False, test_size=test_size)
        # Summarize distribution
        logger.info("Start Grid search")
        CV_rfc = GridSearchCV(
            estimator=p.estimator,
            param_grid=p.PARAMETER_GRID,
            cv=cv,
            n_jobs=n_jobs,
            scoring=scoring,
            refit='precision',
            verbose=1,
        )
        CV_rfc.fit(X_train, y_train)
        logger.info("End Grid search")

        # Take the fitted ensemble with tuned hyperparameters
        clf = CV_rfc.best_estimator_
        best_score = CV_rfc.best_score_
        best_params = CV_rfc.best_params_

        # Plot learning curve for the classifier
        est = p.estimator
        est.set_params(**best_params)
        plot_learning_curve(est, "{} - Learning curves".format(_sym), X_train, y_train)
        curve_path = '{}{}_learning_curve.png'.format(reports_dir, _sym)
        plt.savefig(curve_path)
        plt.close()

        # Test ensemble's performance on training and test sets
        predictions1 = clf.predict(X_train)
        train_report = classification_report(y_train, predictions1, output_dict=True)
        logger.info("Classification report on train set:\n{}".format(classification_report(y_train, predictions1)))
        predictions2 = clf.predict(X_test)
        test_report = classification_report(y_test, predictions2, output_dict=True)
        logger.info("Classification report on test set\n{}".format(classification_report(y_test, predictions2)))

        report = {
            'training_set': {
                'features':X_train.shape[1],
                'records':X_train.shape[0],
                'class_distribution': get_class_distribution(y_train),
                'classification_report': train_report,
                'accuracy': accuracy_score(y_train, predictions1),
                'mse': mean_squared_error(y_train, predictions1),
                'precision': precision_score(y_train, predictions1),
                'recall': recall_score(y_train, predictions1),
                'f1': f1_score(y_train, predictions1),
                'y_true':[y for y in y_train],
                'y_pred':[y for y in predictions1]
            },
            'test_set': {
                'features':X_test.shape[1],
                'records':X_test.shape[0],
                'class_distribution':get_class_distribution(y_test),
                'classification_report': test_report,
                'accuracy': accuracy_score(y_test, predictions2),
                'precision': precision_score(y_test, predictions2),
                'mse': mean_squared_error(y_test, predictions2),
                'recall': recall_score(y_test, predictions2),
                'f1': f1_score(y_test, predictions2),
                'y_true': [y for y in y_test],
                'y_pred': [y for y in predictions2]
            }
        }
        # If the classifier has a feature_importances attribute, save it in the report
        feature_importances = None
        if hasattr(clf, 'feature_importances_'):
            feature_importances = clf.feature_importances_
        elif hasattr(clf, 'named_steps') and hasattr(clf.named_steps, 'c') and hasattr(clf.named_steps.c, 'feature_importances_'):
            feature_importances = clf.named_steps.c.feature_importances_
        if feature_importances is not None:
            importances = {features.columns[i]: v for i, v in enumerate(feature_importances)}
            labeled = {str(k): float(v) for k, v in sorted(importances.items(), key=lambda item: -item[1])}
            report['feature_importances'] = labeled
        if hasattr(clf, 'ranking_'):
            report['feature_rank'] = {features.columns[i]: s for i, s in enumerate(clf.ranking_)}
        if hasattr(clf, 'support_'):
            report['feature_support'] = [features.columns[i] for i, s in enumerate(clf.support_) if s]
        train_dist = ['\t\tClass {}:\t{}\t({}%%)'.format(k, d['count'], d['pct']) for k, d in get_class_distribution(y_train).items()]
        test_dist = ['\t\tClass {}:\t{}\t({}%%)'.format(k, d['count'], d['pct']) for k, d in get_class_distribution(y_test).items()]

        logger.info('Model evaluation: \n'
              '== Training set ==\n'
              '\t # Features: {} | # Records: {}\n '
              '\tClass distribution:\n{}\n'
              '\tAccuracy: {}\n'
              '\tPrecision: {}\n'
              '\tMSE: {}\n' \
              '\tRecall: {}\n' \
              '\tF1: {}\n' \
              '== Test set ==\n'
              '\t # Features: {} | # Records: {}\n '
              '\tClass distribution:\n{}\n'
              '\tAccuracy: {}\n'
              '\tPrecision: {}\n'
              '\tMSE: {}\n' \
              '\tRecall: {}\n' \
              '\tF1: {}\n' \
              .format(X_train.shape[1], X_train.shape[0], '\n'.join(train_dist),
                      report['training_set']['accuracy'], report['training_set']['precision'], report['training_set']['mse'],
                      report['training_set']['recall'], report['training_set']['f1'],
                      X_test.shape[1], X_test.shape[0], '\n'.join(test_dist),
                      report['test_set']['accuracy'], report['test_set']['precision'], report['test_set']['mse'],
                      report['test_set']['recall'], report['test_set']['f1']
                      )
        )

        # Save a pickle dump of the model
        model_path = '{}{}.p'.format(models_dir, _sym)
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)
        # Save the model's parameters
        params_path = '{}{}_parameters.json'.format(models_dir, _sym)
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        # Save the report for this model
        report_path = '{}{}.json'.format(reports_dir, _sym)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        # Update the experiment's index with the new results, and save it
        experiment_index[_sym] = {
            'model':model_path,
            'params':params_path,
            'report':report_path
        }
        with open(experiment_index_file, 'w') as f:
            json.dump(experiment_index, f, indent=4)
        logger.info("--- {} end ---".format(_sym))
    return experiment_index

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
    parser.add_argument('--expanding-window',dest='expanding_window', nargs='?', default=False, help="Use TimeSeriesSplit instead of StratifiedKFold as cross validation provider")
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
        expanding_window=args.expanding_window
    )
