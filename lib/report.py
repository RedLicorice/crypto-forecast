import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lib.plot import plot_learning_curve
from lib.dataset import get_class_distribution
from scikitplot.metrics import plot_roc, plot_precision_recall
from sklearn.metrics import plot_confusion_matrix, make_scorer, classification_report
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, recall_score, precision_score
from lib.log import logger
import pickle, json


class ReportCollection:
    def __init__(self, dataset, pipeline, experiment):
        self.dataset = dataset
        self.pipeline = pipeline
        self.experiment = experiment
        self.reports_dir = './results/{}_{}_{}/'.format(dataset, pipeline, experiment)
        self.index_path = './results/{}_{}_{}/index.json'.format(dataset, pipeline, experiment)
        self.index = {}

    def save(self):
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=4)

    def load(self):
        with open(self.index_path, 'r') as f:
            self.index = json.load(f)

    def add_report(self, report):
        self.index[report.symbol] = report.save(self.reports_dir)

    def get_reports(self):
        result = {}
        for symbol, paths in self.index.items():
            with open(paths['full_report'], 'rb') as f:
                result[symbol] = pickle.load(f)
        return result

    def get_summaries(self):
        result = {}
        for symbol, paths in self.index.items():
            with open(paths['report'], 'r') as f:
                result[symbol] = json.load(f)
        return result

    def get_metrics_df(self):
        data = {}
        for _sym, report in self.get_summaries():
            if not report:
                continue
            train_accuracy = report['training_set']['accuracy']
            train_precision = report['training_set']['precision']
            train_mse = report['training_set']['mse']
            test_accuracy = report['test_set']['accuracy']
            test_precision = report['test_set']['precision']
            test_mse = report['test_set']['mse']
            data[_sym] = [train_accuracy, test_accuracy, train_precision, test_precision, train_mse,  test_mse]
        return pd.DataFrame.from_dict(data, orient='index', columns=['train_accuracy', 'test_accuracy','train_precision','test_precision','train_mse','test_mse'])



class Report():
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __init__(self, symbol, target, cv):
        self.symbol = symbol
        self.target = target
        self.cv = cv

    def set_close(self, close):
        self.closing_price = close

    def set_dataset_columns(self, labels):
        self.column_labels = [c for c in labels]

    def set_train_dataset(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def set_test_dataset(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def set_model(self, model):
        self.model = model

    def set_params(self, params):
        self.params = params

    def set_cv(self, fit_estimator, best_score, cv_results, evaluate=True):
        self.cv_estimator = fit_estimator
        self.cv_best_score = best_score
        self.cv_results = cv_results
        #
        # If the classifier has a feature_importances attribute, save it in the report
        #
        feature_importances = None
        if hasattr(fit_estimator, 'feature_importances_'):
            feature_importances = fit_estimator.feature_importances_
        elif hasattr(fit_estimator, 'named_steps') and hasattr(fit_estimator.named_steps, 'c') and hasattr(fit_estimator.named_steps.c, 'feature_importances_'):
            feature_importances = fit_estimator.named_steps.c.feature_importances_
        if feature_importances is not None:
            importances = {self.column_labels[i]: v for i, v in enumerate(feature_importances)}
            labeled = {str(k): float(v) for k, v in sorted(importances.items(), key=lambda item: -item[1])}
            self.feature_importances = labeled
        if hasattr(fit_estimator, 'ranking_'):
            self.feature_rank = {self.column_labels[i]: s for i, s in enumerate(fit_estimator.ranking_)}
        if hasattr(fit_estimator, 'support_'):
            self.feature_support = [self.column_labels[i] for i, s in enumerate(fit_estimator.support_) if s]
        if evaluate:
            self.evaluate_cv()

    def evaluate_cv(self):
        #
        # Test ensemble's performance on training and test sets
        #
        predictions1 = self.cv_estimator.predict(self.X_train)
        train_report = classification_report(self.y_train, predictions1, output_dict=True)
        logger.info("Classification report on train set:\n{}".format(classification_report(self.y_train, predictions1)))
        predictions2 = self.cv_estimator.predict(self.X_test)
        test_report = classification_report(self.y_test, predictions2, output_dict=True)
        logger.info("Classification report on test set\n{}".format(classification_report(self.y_test, predictions2)))

        report = {
            'training_set': {
                'features': self.X_train.shape[1],
                'records': self.X_train.shape[0],
                # 'class_distribution': get_class_distribution(y_train),
                'classification_report': train_report,
                'accuracy': accuracy_score(self.y_train, predictions1),
                'mse': mean_squared_error(self.y_train, predictions1),
                'precision': precision_score(self.y_train, predictions1, average='micro'),
                'recall': recall_score(self.y_train, predictions1, average='micro'),
                'f1': f1_score(self.y_train, predictions1, average='micro'),
                # 'y_true':[y for y in y_train],
                # 'y_pred':[y for y in predictions1]
            },
            'test_set': {
                'features': self.X_test.shape[1],
                'records': self.X_test.shape[0],
                # 'class_distribution':get_class_distribution(y_test),
                'classification_report': test_report,
                'accuracy': accuracy_score(self.y_test, predictions2),
                'precision': precision_score(self.y_test, predictions2, average='micro'),
                'mse': mean_squared_error(self.y_test, predictions2),
                'recall': recall_score(self.y_test, predictions2, average='micro'),
                'f1': f1_score(self.y_test, predictions2, average='micro'),
                # 'y_true': [y for y in y_test],
                # 'y_pred': [y for y in predictions2]
            }
        }
        train_dist = ['\t\tClass {}:\t{}\t({}%%)'.format(k, d['count'], d['pct']) for k, d in
                      get_class_distribution(self.y_train).items()]
        test_dist = ['\t\tClass {}:\t{}\t({}%%)'.format(k, d['count'], d['pct']) for k, d in
                     get_class_distribution(self.y_test).items()]
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
                    .format(self.X_train.shape[1], self.X_train.shape[0], '\n'.join(train_dist),
                            report['training_set']['accuracy'], report['training_set']['precision'],
                            report['training_set']['mse'],
                            report['training_set']['recall'], report['training_set']['f1'],
                            self.X_test.shape[1], self.X_test.shape[0], '\n'.join(test_dist),
                            report['test_set']['accuracy'], report['test_set']['precision'], report['test_set']['mse'],
                            report['test_set']['recall'], report['test_set']['f1']
                            )
                    )
        if self.feature_importances:
            report['feature_importances'] = self.feature_importances
        self.cv_report = report
        return report

    def plot_learning_curve(self, save_dir):
        # Plot learning curve for the classifier
        est = self.model
        est.set_params(**self.params)

        _, axes = plt.subplots(3, 3, figsize=(20, 12), dpi=200, constrained_layout=True)
        # plt.tight_layout()
        _train_ax = [axes[0][0], axes[0][1], axes[0][2]]
        plot_learning_curve(est, "{} - Learning curves (Train)".format(self.symbol), self.X_train, self.y_train, axes=_train_ax, cv=self.cv)

        n_classes = np.unique(self.y_test).shape[0]
        if hasattr(self.cv_estimator, 'predict_proba'):
            y_train_proba = self.cv_estimator.predict_proba(self.X_train)
            axes[1][0].set_title("{} - ROC (Train)".format(self.symbol))
            plot_roc(self.y_train, y_train_proba, n_classes, ax=axes[1][0])
            axes[1][1].set_title("{} - Precision/Recall (Train)".format(self.symbol))
            plot_precision_recall(self.y_train, y_train_proba, ax=axes[1][1])
        axes[1][2].set_title("{} - Confusion matrix (Train)".format(self.symbol))
        plot_confusion_matrix(self.cv_estimator, self.X_train, self.y_train, cmap='Blues', ax=axes[1][2])
        if hasattr(self.cv_estimator, 'predict_proba'):
            y_test_proba = self.cv_estimator.predict_proba(self.X_test)
            axes[2][0].set_title("{} - ROC (Test)".format(self.symbol))
            plot_roc(self.y_test, y_test_proba, ax=axes[2][0])
            axes[2][1].set_title("{} - Precision/Recall (Test)".format(self.symbol))
            plot_precision_recall(self.y_test, y_test_proba, ax=axes[2][1])
        axes[2][2].set_title("{} - Confusion matrix (Test)".format(self.symbol))
        plot_confusion_matrix(self.cv_estimator, self.X_test, self.y_test, cmap='Oranges', ax=axes[2][2])

        curve_path = '{}{}_learning_curve.png'.format(save_dir, self.symbol)
        plt.savefig(curve_path)
        plt.close()

    def save(self, directory):
        self.plot_learning_curve(directory)
        # Save a pickle dump of the model
        model_path = '{}{}_model.p'.format(directory, self.symbol)
        with open(model_path, 'wb') as f:
            pickle.dump(self.cv_estimator, f)
        # Save the model's parameters
        params_path = '{}{}_parameters.json'.format(directory, self.symbol)
        with open(params_path, 'w') as f:
            json.dump(self.params, f, indent=4)
        # Save the report for this model
        report_path = '{}{}_report.json'.format(directory, self.symbol)
        with open(report_path, 'w') as f:
            json.dump(self.cv_report, f, indent=4)
        # Save a pickle dump of this report
        full_report = '{}{}_report.p'.format(directory, self.symbol)
        with open(full_report, 'wb') as f:
            pickle.dump(self, f)
        return {
                'model':model_path,
                'params':params_path,
                'report':report_path,
                'full_report':full_report
            }