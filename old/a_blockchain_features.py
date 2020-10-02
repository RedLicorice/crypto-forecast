import logging
from build_datasets_old import load_dataset
from lib.log import logger
from lib.dataset import target_price_variation, target_discrete_price_variation, target_binned_price_variation, discretize_ta_features
import pandas as pd
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, RobustScaler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
import json

def main():
    index = load_dataset('all_merged', return_index=True)
    for _sym, data in index.items():

        features = pd.read_csv(data['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        # Replace nan with infinity so that it can later be imputed to a finite value
        features = features.replace([np.inf, -np.inf], np.nan)

        # Derive target classes from closing price
        target_pct = target_price_variation(features['close'])
        target = target_binned_price_variation(target_pct, n_bins=2)
        # target = target_discrete_price_variation(target_pct)

        print("--- end ---")

if __name__ == '__main__':
    logger.setup(
        filename='../blockchain_features.log',
        filemode='w',
        root_level=logging.DEBUG,
        log_level=logging.DEBUG,
        logger='blockchain_features'
    )
    main()
