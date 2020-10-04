from lib.models import migrate, Base, Job
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker, scoped_session
from lib.log import logger
from build_model import build_model
import pandas as pd
import json, os
import argparse
import numpy as np

DBSession = None

PIPELINES = [
    # Boosting
    #'adaboost_decisiontree', # We already have xgboost
    'plain_xgboost',
    #'plain_dart_xgboost', # Much slower, no improvements
    # 'pca_kbest_xgboost', # No improvement w.r.t xgboost
    # 'pca_xgboost', # No improvement w.r.t xgboost

    # Bagging
    #'bagging_decisiontree', # Gets stuck
    'bagging_linear_svc',
    'bagging_poly_svc',
    'bagging_rbf_svc',

    # Plain ML
    'plain_knn',
    'plain_linear_svc',
    'plain_poly_svc',
    'plain_rbf_svc',
    'plain_mlp',
    'plain_mnb',
    'plain_randomforest',
    #'smote_undersampler_svc'
]

DATASETS = [
    'improved',
    'atsa',
    # 'all_merged.index_faceted'
]

TARGETS = [
    'class',
    'binary',
    'bin',
    'binary_bin'
]

def bench_models(benchmark_name):
    session = DBSession()
    jobs = session.query(Job).filter(Job.status == False).all()
    for j in jobs:
        logger.info("Running experiment: {} {} {} {}".format(j.id, j.dataset, j.pipeline, j.target, benchmark_name))
        experiment = build_model(j.dataset, j.pipeline, "{}_{}".format(benchmark_name, j.target), scoring='precision', use_target=j.target)
        data = {}
        for _sym, results in experiment.items():
            with open(results['report'], 'r') as f:
                report = json.load(f)
            if not report:
                continue
            train_accuracy = report['training_set']['accuracy']
            train_precision = report['training_set']['precision']
            train_mse = report['training_set']['mse']
            test_accuracy = report['test_set']['accuracy']
            test_precision = report['test_set']['precision']
            test_mse = report['test_set']['mse']
            data[_sym] = [train_accuracy, test_accuracy, train_precision, test_precision, train_mse,  test_mse]
        pipe_df = pd.DataFrame.from_dict(data, orient='index', columns=['train_accuracy', 'test_accuracy','train_precision','test_precision','train_mse','test_mse'])

        pipe_df.to_csv('./benchmarks/{}/{}_{}_{}.csv'.format(benchmark_name, j.dataset, j.pipeline, j.target), sep=',',
                       encoding='utf-8', index=True,
                       index_label='Symbol')
        pipe_df.to_excel('./benchmarks/{}/{}_{}_{}.xlsx'.format(benchmark_name, j.dataset, j.pipeline, j.target),
                         index=True, index_label='Symbol')
        j.dataframe = './benchmarks/{}/{}_{}_{}.csv'.format(benchmark_name, j.dataset, j.pipeline, j.target)
        j.status = True
        session.commit()
        logger.info("End experiment: {} {} {} {}".format(j.id, j.dataset, j.pipeline, j.target, benchmark_name))

    for d in DATASETS:
        for t in TARGETS:
            jobs = session.query(Job).filter(and_(Job.status == True, Job.dataset == d, Job.target == t)).all()
            if jobs:
                print ("Merging results Dataset: {} Target: {} ({} jobs)".format(d, t, len(jobs)))
                result = {}
                for j in jobs:
                    result[j.pipeline] = pd.read_csv(j.dataframe, sep=',', encoding='utf-8', index_col='Symbol')
                stacked = pd.concat(result, axis=1)
                stacked.to_csv('./benchmarks/{}/{}_{}_merged.csv'.format(benchmark_name, d, t), sep=',',
                               encoding='utf-8', index=True,
                               index_label='Symbol')
                stacked.to_excel('./benchmarks/{}/{}_{}_merged.xlsx'.format(benchmark_name, d, t),
                                 index=True, index_label='Symbol')


def make_jobs():
    session = DBSession()
    count = 0
    for t in TARGETS:
        for p in PIPELINES:
            for d in DATASETS:
                job = Job(
                    dataset=d,
                    pipeline=p,
                    target=t,
                    status=False
                )
                session.add(job)
                count += 1
    session.commit()
    logger.info("Created {} jobs".format(count))

if __name__ == '__main__':
    np.random.seed(0)
    parser = argparse.ArgumentParser(
        description='Build and tune models, collect results')
    parser.add_argument('-n', dest='name', nargs='?', default='benchmark',
                        help="Name for the current benchmark")  # nargs='?', default='all_merged.index_improved',
    args = parser.parse_args()
    os.makedirs('./benchmarks/{}/'.format(args.name), exist_ok=True)
    db_file = './benchmarks/{}/status.db'.format(args.name)
    engine = create_engine('sqlite:///' + db_file)
    Base.metadata.bind = engine
    session_factory = sessionmaker(bind=engine)
    DBSession = scoped_session(session_factory)
    if migrate(db_file): # Create status database and tables
        make_jobs()
    bench_models(args.name)