from lib.dataset.utils import *
from lib.dataset import builder
from collections import Counter
import pandas as pd
import json

def load_preprocessed(source):
    index_path = './data/preprocessed/{}/index.json'.format(source)
    index = {}
    with open(index_path, 'r') as f:
        index = json.load(f)
    return index

def load_dataset(dataset, return_index=False, index_name='index'):
    index_path = './data/datasets/{}/{}.json'.format(dataset, index_name)
    index = {}
    with open(index_path, 'r') as f:
        index = json.load(f)
    if return_index:
        return index
    return {sym: (pd.read_csv(files['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True),
                pd.read_csv(files['target_csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True))
            for sym, files in index.items()}

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
        result[k] = {'count':v, 'pct':per}
    return result