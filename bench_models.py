from build_model import build_model
import pandas as pd
import json, os
import argparse
import numpy as np

PIPELINES = [
    'adaboost_decisiontree',
    'bagging_decisiontree',
    'bagging_linear_svc',
    'bagging_poly_svc',
    'bagging_rbf_svc',
    'pca_xgboost',
    'plain_knn',
    'plain_linear_svc',
    'plain_mlp',
    'plain_mnb',
    'plain_poly_svc',
    'plain_randomforest',
    'plain_rbf_svc',
    'plain_xgboost',
    #'rfe_xgboost',
    #'smote_undersampler_svc'
]

DATASETS = [
    #'all_merged.index_improved',
    'all_merged.index_atsa'
]

def bench_models(benchmark_name):
    os.makedirs('./benchmarks/{}/'.format(benchmark_name), exist_ok=True)
    for ds in DATASETS:
        result = {}
        for pipe in PIPELINES:
            experiment = build_model(ds, pipe, benchmark_name)
            data = {}
            for _sym, results in experiment.items():
                with open(results['report'], 'r') as f:
                    report = json.load(f)
                if not report:
                    continue
                train_accuracy = report['training_set']['accuracy']
                train_precision = report['training_set']['precision']
                train_mse = report['training_set']['precision']

                test_accuracy = report['test_set']['accuracy']
                test_precision = report['test_set']['precision']
                test_mse = report['test_set']['precision']

                data[_sym] = [train_accuracy, test_accuracy, train_precision, test_precision, train_mse,  test_mse]
            pipe_df = pd.DataFrame.from_dict(data, orient='index', columns=['train_accuracy', 'test_accuracy','train_precision','test_precision','train_mse','test_mse'])

            pipe_df.to_csv('./benchmarks/{}/{}__{}.csv'.format(benchmark_name, ds, pipe), sep=',',
                           encoding='utf-8', index=True,
                           index_label='Symbol')
            pipe_df.to_excel('./benchmarks/{}/{}__{}.xlsx'.format(benchmark_name, ds, pipe),
                             index=True, index_label='Symbol')
            result[pipe] = pipe_df
        stacked = pd.concat(result, axis=1)
        stacked.to_csv('./benchmarks/{}/{}_merged.csv'.format(benchmark_name, ds), sep=',',
                       encoding='utf-8', index=True,
                       index_label='Symbol')
        stacked.to_excel('./benchmarks/{}/{}_merged.xlsx'.format(benchmark_name, ds),
                         index=True, index_label='Symbol')

if __name__ == '__main__':
    np.random.seed(0)
    parser = argparse.ArgumentParser(
        description='Build and tune models, collect results')
    parser.add_argument('-n', dest='name', nargs='?', default='bench_models',
                        help="Name for the current benchmark")  # nargs='?', default='all_merged.index_improved',
    args = parser.parse_args()
    bench_models(args.name)