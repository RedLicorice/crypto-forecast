from lib.dataset import load_preprocessed
from lib.technical_indicators import *
import pandas as pd
from matplotlib import pyplot as plt


class ReactiveStrategy():
    def __init__(self):
        pass

    def open(self, symbol):
        index = load_preprocessed('ohlcv')
        if not symbol in index:
            raise Exception("Symbol {} is not in OHLCV index!".format(symbol))
        csv_path = index[symbol]['csv']
        self.ohlcv = pd.read_csv(csv_path, sep=',', encoding='utf-8', index_col='Date', parse_dates=True)

    def macd_signal(self, day):
        macd = moving_average_convergence_divergence(self.ohlcv.close, 12, 26)
        signal = exponential_moving_average(macd, 9)
        df = pd.DataFrame(index=self.ohlcv.index)
        df['macd'] = macd
        df['signal'] = signal
        df['label'] = df.macd - df.signal


if __name__ == '__main__':
    r = ReactiveStrategy()
    r.open('BTC')
    r.macd_signal('2018-01-01')