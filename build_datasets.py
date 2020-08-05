import logging
from lib.log import logger
import pandas as pd
from lib.plotter import correlation
import lib.dataset as builder
import json, os

def load_preprocessed(source):
    index_path = './data/preprocessed/{}/index.json'.format(source)
    index = {}
    with open(index_path, 'r') as f:
        index = json.load(f)
    return index

def load_dataset(dataset):
    index_path = './data/datasets/{}/index.json'.format(dataset)
    index = {}
    with open(index_path, 'r') as f:
        index = json.load(f)
    return index

def make_features(source, dataset):
    index = load_preprocessed(source)
    for _sym, files in index.items():
        df = pd.read_csv(files['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        diff = builder.difference(df)
        pct = builder.pct_change(df)

