import logging
from build_datasets_old import load_dataset
from lib.log import logger
from lib.dataset import target_price_variation, target_discrete_price_variation, target_binned_price_variation, discretize_ta_features
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
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

RANDOMFOREST_PARAM_GRID = {
    'c__n_estimators': [100, 200, 500],
    'i__strategy': ['mean'],  # 'median', 'most_frequent', 'constant'
    'c__criterion': ['gini'],  # , 'entropy'],
    'c__max_depth': [4, 8],
    'c__min_samples_split': [2],
    'c__min_samples_leaf': [1, 0.1, 0.2],
    'c__max_features': ['auto', 'log2', 0.1, 0.2, 0.6],  # 'sqrt',
    'c__class_weight': [None],#, 'balanced', 'balanced_subsample'
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
    resultFile = './data/datasets/all_merged/estimators/randomforest_sfm_imp_hyperparameters.json'
    estFile = './data/datasets/all_merged/estimators/randomforest_sfm_imp_{}.p'
    hyperparameters = {}
    if os.path.exists(resultFile):
        with open(resultFile, 'r') as f:
            hyperparameters = json.load(f)
    for _sym, data in index.items():
        features = pd.read_csv(data['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        target = pd.read_csv(data['target_csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        # Replace nan with infinity so that it can later be imputed to a finite value
        features = features.replace([np.inf, -np.inf], np.nan)
        # Derive target classes from closing price
        # target_pct = target_price_variation(features['close'])
        # target = target_binned_price_variation(target_pct, n_bins=2)
        target = target.loc[features.first_valid_index():features.last_valid_index()]['binary_bin']
        # target = target_discrete_price_variation(target_pct)

        # Split data in train and blind test set with 70:30 ratio,
        #  most ML models don't take sequentiality into account, but our pipeline
        #  uses a SimpleImputer with mean strategy, so it's best not to shuffle the data.
        X_train, X_test, y_train, y_test = train_test_split(features.values, target.values, shuffle=False, test_size=0.3)
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
            ('i', SimpleImputer()), # Replace nan's with the median value between previous and next observation
            ('s', RobustScaler()), # Scale data in order to center it and increase robustness against noise and outliers
            #('k', SelectKBest()), # Select top 10 best features
            #('u', RandomUnderSampler()),
            ('c', RandomForestClassifier()),
        ])
        # Skip grid search if there are already hyperparameters for this set
        if _sym in hyperparameters and os.path.exists(hyperparameters[_sym]['estimator']):
            logger.info('{} already exists.'.format(_sym))
        else:
            # Perform hyperparameter tuning of the ensemble with 5-fold cross validation
            logger.info("Start Grid search")
            CV_rfc = GridSearchCV(
                estimator=pipeline,
                param_grid=RANDOMFOREST_PARAM_GRID,
                cv=5,
                n_jobs=4,
                #scoring='neg_mean_squared_error',
                scoring='roc_auc',
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
                'test_score_7': accuracy_score(y_test[0:7], predictions2[0:7]),
                'test_mse_7': mean_squared_error(y_test[0:7], predictions2[0:7]),
                'train_report': train_report,
                'test_report': test_report,
                'cv_best_score': CV_rfc.best_score_,
                # 'cv_results': CV_rfc.cv_results_,
                'cv_bestparams': CV_rfc.best_params_,
            }
            print(json.dumps(stats, indent=4))
            with open(estFile.format(_sym), 'wb') as f:
                pickle.dump(clf, f)

            input = pd.DataFrame(X_train)#.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').replace(np.nan, 0)
            imp = SimpleImputer()
            input = imp.fit_transform(input)
            sca = RobustScaler()
            input = sca.fit_transform(input)
            sfm = SelectFromModel(estimator=clf.named_steps.c, prefit=True)
            sfm.transform(input)
            sup = sfm.get_support()
            sel_features = [c for c, p in zip(features.columns, sup) if p]
            importances = {features.columns[i]: v for i, v in enumerate(clf.named_steps.c.feature_importances_)}
            labeled_importances = {str(k): v for k, v in sorted(importances.items(), key=lambda item: -item[1])}
            hyperparameters[_sym] = {
                'estimator': estFile.format(_sym),
                'stats': stats,
                'features': sel_features,
                'feature_importances':labeled_importances
            }
            with open(resultFile, 'w') as f: # Save results at every update
                json.dump(hyperparameters, f, indent=4)
            print("--- end ---")

if __name__ == '__main__':
    logger.setup(
        filename='../dataset_info_randomforest_sfm.log',
        filemode='w',
        root_level=logging.DEBUG,
        log_level=logging.DEBUG,
        logger='dataset_info_randomforest_sfm'
    )
    main()
