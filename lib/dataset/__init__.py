from lib.dataset.utils import *
from lib.dataset import builder
from collections import Counter
import pandas as pd
import json
import os

def load_preprocessed(source):
    index_path = './data/preprocessed/{}/index.json'.format(source)
    index = {}
    with open(index_path, 'r') as f:
        index = json.load(f)
    return index

def load_dataset(dataset, return_index=False, index_name='index'):
    index_path = './data/datasets/{}/index.json'.format(dataset, index_name)
    index = {}
    with open(index_path, 'r') as f:
        index = json.load(f)
    if return_index:
        return index
    return {sym: (pd.read_csv(files['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True),
                pd.read_csv(files['target_csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True))
            for sym, files in index.items()}

def save_symbol_dataset(dataset, symbol, df, **kwargs):
    # Split index if dataset contains it
    # Compose file paths
    features_csv_path = 'data/datasets/{}/csv/{}.csv'.format(dataset, symbol.lower())
    features_xls_path = 'data/datasets/{}/excel/{}.xlsx'.format(dataset, symbol.lower())
    target_csv_path = 'data/datasets/{}/csv/{}_target.csv'.format(dataset, symbol.lower())
    target_xls_path = 'data/datasets/{}/excel/{}_target.xlsx'.format(dataset, symbol.lower())
    index_path = 'data/datasets/{}/index.json'.format(dataset)
    # Make sure directories exist
    for p in [features_csv_path,features_xls_path,target_csv_path,target_xls_path,index_path]:
        dir = os.path.dirname(p)
        os.makedirs(dir, exist_ok=True)
    # Save features
    df.to_csv(features_csv_path, sep=',', encoding='utf-8', index=True, index_label='Date')
    df.to_excel(features_xls_path, index=True, index_label='Date')
    # If a target is provided, use it
    if 'target' in kwargs:
        target = kwargs.get('target')
        target.to_csv(target_csv_path, sep=',', encoding='utf-8', index=True, index_label='Date')
        target.to_excel(target_xls_path, index=True, index_label='Date')
    # Update dataset index
    index = {}
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index = json.load(f)
    index[symbol] = {
        'csv': features_csv_path,
        'xls': features_xls_path,
        'target_csv': target_csv_path,
        'target_xls': target_xls_path,
        'features': kwargs.get('feature_groups') if 'feature_groups' in kwargs else {'all':[c for c in df.columns]}
    }
    with open(index_path, 'w') as f:
        json.dump(index, f, sort_keys=True, indent=4)


def print_class_distribution(y):
    counter = Counter(y)
    for k, v in counter.items():
        per = v / len(y) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

def get_class_distribution(y):
    counter = Counter(y)
    result = {}
    for k, v in counter.items():
        per = v / len(y) * 100
        result[int(k)] = {'count':v, 'pct':per}
    return result