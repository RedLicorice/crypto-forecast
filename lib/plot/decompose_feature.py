from lib.dataset import is_stationary
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def decompose_feature(name, series: pd.Series):
    series = series.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").dropna()
    if series.shape[0] < 14:
        print("Cannot decompose {} ({} records)".format(name, series.shape[0]))
        return None
    decompose_result = seasonal_decompose(series)
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(32, 18))
    fig.suptitle('{} {}'.format(name, "[Stationary]" if is_stationary(series) else "[Non-Stationary]"), fontsize=16)
    axes[0].set_title(label='Observed')
    axes[1].set_title(label='Trend')
    axes[2].set_title(label='Seasonal')
    axes[3].set_title(label='Residual')

    axes[0].plot(series.index, decompose_result.observed)
    axes[1].plot(series.index, decompose_result.trend)
    axes[2].plot(series.index, decompose_result.seasonal)
    axes[3].plot(series.index, decompose_result.resid)
    plot_acf(x=series.values, ax=axes[4], title="ACF")
    #plot_pacf(x=series.values, ax=axes[5], title="PACF")