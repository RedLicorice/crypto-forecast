import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def to_discrete_single(values, threshold):
    def _to_discrete(x, threshold):
        if np.isnan(x):
            return -1
        if x < threshold:
            return 0
        return 1

    fun = np.vectorize(_to_discrete)
    return fun(values, threshold)


def to_discrete_double(values, threshold_lo=0.01, threshold_hi=0.01):
    def _to_discrete(x, threshold_lo, threshold_hi):
        if np.isnan(x):
            return -1
        if x <= threshold_lo:
            return 0
        elif threshold_lo < x < threshold_hi:
            return 1
        else:
            return 2
    fun = np.vectorize(_to_discrete)
    return fun(values, threshold_lo, threshold_hi)

# Returns the ratio [missing values] / [total values]
def get_missing_ratio(series: pd.Series):
    null_values_count = series.replace([np.inf, -np.inf], np.nan).isna().sum()
    return null_values_count / max(1, series.count())

# Returns the ratio [unique non-null values] / [total values]
def get_idness(series: pd.Series, round_decimals=2):
    unique_values_count = series.round(decimals=round_decimals).nunique(dropna=True)
    return unique_values_count / max(1, series.count())

# Returns the ratio [occurrencies of most frequent value] / [total values]
def get_stability(series: pd.Series, round_decimals=2):
    if not series.dropna().count():
        return 1
    most_frequent_value_count = series.round(decimals=round_decimals).value_counts().iloc[0]
    return most_frequent_value_count / max(1, series.count())

def feature_quality_filter(df: pd.DataFrame, missing_threshold=0.3, idness_threshold=0.95, stability_threshold=0.8):
    drop_missing = []
    drop_stability = []
    drop_idness = []
    for c in df.columns:
        missing_ratio = get_missing_ratio(df[c])
        idness = get_idness(df[c])
        stability = get_stability(df[c])
        print("Feature: {} \n\tMissing: {}%\n\tID-Like: {}%\n\tStability: {}%".format(
            c, round(missing_ratio, 2), round(idness, 2), round(stability, 2)
        ))
        if missing_ratio > missing_threshold:
            drop_missing.append(c)
        if stability > stability_threshold:
            drop_stability.append(c)
        if idness > idness_threshold:
            drop_idness.append(c)
    print("Dropping features with too many missing values [{}]:\n{}".format(len(drop_missing), drop_missing))
    print("Dropping features with too many distinct values [{}]:\n{}".format(len(drop_idness), drop_idness))
    print("Dropping features with too few distinct values [{}]:\n{}".format(len(drop_stability), drop_stability))
    drop = drop_missing + drop_stability + drop_idness
    return df.drop(labels=drop, axis=1)

def is_stationary(series: pd.Series, log=False):
    adfstat, adf_pvalue, usedlag, nobs, adf_critvalues, icbest = adfuller(series.values, autolag='AIC')
    if log:
        print("==== ADF ====")
        print("1. ADF : ", adfstat)
        print("2. P-Value : ", adf_pvalue)
        print("3. Num Of Lags : ", usedlag)
        print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", nobs)
        print("5. Critical Values :")
        for key, val in adf_critvalues.items():
            print("\t", key, ": ", val)
    # P-Value > 0.05 means the time series is not stationary
    return adf_pvalue < 0.05