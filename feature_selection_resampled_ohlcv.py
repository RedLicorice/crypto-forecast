import logging
from lib.log import logger
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPClassifier
from genetic_selection import GeneticSelectionCV
from sklearn.feature_selection import SelectKBest, chi2
from matplotlib import pyplot as plt
from multiprocessing import freeze_support
from lib.utils import scale
import numpy as np
import json

INTERACTIVE_FIGURE = False
SYMBOLS = [
    'ADA',
    'BCH',
    'BNB',
    'BTC',
    'BTG',
    'DASH',
    'DOGE',
    'EOS',
    'ETC',
    'ETH',
    'LTC',
    'LINK',
    'NEO',
    'QTUM',
    'TRX',
    'USDT',
    'VEN',
    'WAVES',
    'XEM',
    'XMR',
    'XRP',
    'ZEC',
    'ZRX'
]

def main():
    result = {}
    for _sym in SYMBOLS:
        dataset = 'data/result/datasets/csv/{}.csv'.format(_sym)
        df = pd.read_csv(dataset, sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        X = df[df.columns.difference(['target','target_pct', 'target_label'])]
        y = df['target']
        #print("======"+_sym+"======")
        #print(X.info())

        # Variance Threshold
        sel = VarianceThreshold()
        sel.fit_transform(X)
        sup = sel.get_support()
        X = X[[name for flag, name in zip(sup, X.columns) if flag]]
        ## SelectKBest
        sel = SelectKBest(chi2, k=30)
        sX = scale(X, scaler='minmax')
        sel.fit_transform(sX,y)
        sup = sel.get_support()
        sX = sX[[name for flag, name in zip(sup, sX.columns) if flag]]

        ## Recursive Feature Elimination
        # Create the RFE object and compute a cross-validated score.
        # The "accuracy" scoring is proportional to the number of correct
        # classifications
        # model = SVC(kernel="linear")
        # rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(2), scoring='accuracy', n_jobs=-1, verbose=1)
        # rfecv.fit(X, y)
        # X = X[[name for flag, name in zip(rfecv.support_, X.columns) if flag]]
        ### Genetic
        # estimator = MLPClassifier(**{
        #     'hidden_layer_sizes': (10, 4),
        #     'solver': 'lbfgs',
        #     'learning_rate': 'constant',
        #     'learning_rate_init': 0.001,
        #     'activation': 'logistic'
        # })
        estimator = LogisticRegression(solver="liblinear", multi_class="ovr")
        gscv = GeneticSelectionCV(estimator,
                                      cv=2,
                                      verbose=1,
                                      scoring="accuracy",
                                      max_features=30,
                                      n_population=50,
                                      crossover_proba=0.5,
                                      mutation_proba=0.2,
                                      n_generations=80,
                                      crossover_independent_proba=0.5,
                                      mutation_independent_proba=0.05,
                                      tournament_size=3,
                                      n_gen_no_change=10,
                                      caching=True,
                                      n_jobs=-1)
        gscv = gscv.fit(X, y)
        X = X[[name for flag, name in zip(gscv.support_, X.columns) if flag]]

        #print(X.columns)

        # print("[%s] Optimal number of features : %d Set: %s" % (_sym, rfecv.n_features_, ', '.join(X.columns)))
        # plt.figure()
        # plt.title(_sym + ' SVC RFECV K=2')
        # plt.xlabel("Number of features selected")
        # plt.ylabel("Cross validation score (nb of correct classifications)")
        # plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        # plt.show()

        logger.info("{}: {}".format(_sym, X.columns))
        result[_sym] = {'dataset': dataset, 'columns_genetic_lr_30': [c for c in X.columns], 'columns_kbest_30': [c for c in sX.columns]}
    return result

if __name__ == '__main__':
    freeze_support()
    logger.setup(
        filename='../feature_selection.log',
        filemode='w',
        root_level=logging.DEBUG,
        log_level=logging.DEBUG,
        logger='feature_selection'
    )
    features = main()
    with open('feature_selection_result.json', 'w') as f:
        json.dump(features, f, indent=4, sort_keys=True)
