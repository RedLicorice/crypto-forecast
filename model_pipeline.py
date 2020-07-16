from sklearn.pipeline import Pipeline
import logging
from lib.log import logger
import pandas as pd
from multiprocessing import freeze_support
from sklearn.preprocessing import MinMaxScaler
from genetic_selection import GeneticSelectionCV
from sklearn.linear_model import LogisticRegression
import json

def main(dataset):
    indexFile = 'data/datasets/{}/index.json'.format(dataset)
    resultFile = 'data/datasets/{}/feature_selection.json'.format(dataset)
    with open(indexFile) as f:
        index = json.load(f)

    result = {}
    for _sym, files in index.items():
        params = {
            'estimator': LogisticRegression(**{
                'solver': 'liblinear',
                'multi_class': 'ovr'
            }),
            'cv': 3,
            'verbose': 1,
            'scoring': "accuracy",
            'max_features': 10,
            'n_population': 50,
            'crossover_proba': 0.5,
            'mutation_proba': 0.2,
            'n_generations': 80,
            'crossover_independent_proba': 0.5,
            'mutation_independent_proba': 0.05,
            'tournament_size': 5,
            'n_gen_no_change': 10,
            'caching': True,
            'n_jobs': -1
        }
        pipe = Pipeline([
            ('scaler', MinMaxScaler()),
            ('SVC', GeneticSelectionCV(**params)),
        ])

if __name__ == '__main__':
    freeze_support()
    logger.setup(
        filename='../feature_selection.log',
        filemode='w',
        root_level=logging.DEBUG,
        log_level=logging.DEBUG,
        logger='correlation'
    )
    main()