import pandas as pd
from lib.symbol import DatasetType
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error

class Report:

    def __init__(self, **kwargs):
        self.y = kwargs.get('y')
        self.y_pred = kwargs.get('y_pred')
        self.model = kwargs.get('model') # string
        self.params = kwargs.get('params') # string
        self.errors = kwargs.get('errors')
        self.scores = kwargs.get('scores')
        self.symbol = kwargs.get('symbol')
        self.comparison = kwargs.get('comparison', 'mse')

    def to_dataframe(self):
        result = pd.DataFrame(index=self.y.index)
        result['expected'] = self.y
        result['predicted'] = self.y_pred
        return result

    # For grid search
    def __lt__(self, other):
        if self.comparison == 'mse':
            return self.mse() < other.mse()
        elif self.comparison == 'accuracy':
            return self.accuracy() < other.accuracy()
        elif self.comparison == 'profit':
            return self.profit() < other.profit()

    def __repr__(self):
        return "{}({})".format(self.model, self.params)

    def accuracy(self):
        try:
            return accuracy_score(self.y, self.y_pred)
        except ValueError:
            return -1

    def mse(self):
        return mean_squared_error(self.y, self.y_pred)

    def profit(self, **kwargs):
        start_balance = kwargs.get('start_balance', 10000)
        pos_size = kwargs.get('pos_size', (start_balance*2)/100)
        signals = kwargs.get('signals', self.y_pred)
        if self.symbol is None:
            return
        _, price = self.symbol.get_xy(DatasetType.OHLCV)

        balance = start_balance
        crypto_balance = 0
        history = []
        for i,d in enumerate(self.y.index.date):
            exchange_rate = price.loc[d]
            bag_size = pos_size / exchange_rate
            if signals[i] == 1:
                if crypto_balance > 0:
                    msu = min(bag_size, crypto_balance) # Smallest sellable unit
                    crypto_balance -= msu # Sell msu at market price
                    balance += exchange_rate * msu
            elif signals[i] == 3:
                if balance > start_balance/2:
                    balance -= pos_size
                    crypto_balance += bag_size
            history.append(balance)
        if kwargs.get('history'):
            return history
        return ((balance-start_balance)/start_balance)*100

    def plot_signals(self, **kwargs):
        report = self.to_dataframe()
        ohlc = kwargs.get('ohlc')
        if ohlc is None:
            if self.symbol is None:
                raise ValueError('No source for ohlc data!')
            ohlc = self.symbol.get_dataset(DatasetType.OHLCV)
        ohlc = ohlc[report.index[0]:report.index[-1]]
        ohlc['balance'] = self.profit(history=True)

        ax = ohlc.plot(
            kind='line',
            y=['close'],
            figsize=(12, 10)
        )
        # grid
        xlabels=[d.strftime('%Y-%m-%d %H:%M') for d in ohlc.index]
        ax.set_xticks(xlabels, minor=True)
        ax.set_xticklabels(xlabels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.grid(True, which='minor', axis='x')
        ax.grid(False, which='major', axis='x')
        # Highlight weekends
        sundays = [i for i in range(len(ohlc.index.dayofweek)) if ohlc.index.dayofweek[i] == 6]
        for i in sundays:
            if i + 1 < ohlc.shape[0]:
                ax.axvspan(ohlc.index[i], ohlc.index[i + 1], facecolor='green', edgecolor='none', alpha=.2)
        # expected
        x = report.loc[report['expected'] == 1].index.values
        plt.scatter(x, ohlc.loc[x, 'close'].values - 30, label='skitscat', color='red', s=25, marker="v")
        x = report.loc[report['expected'] == 2].index.values
        plt.scatter(x, ohlc.loc[x, 'close'].values - 30, label='skitscat', color='blue', s=25, marker="o")
        x = report.loc[report['expected'] == 3].index.values
        plt.scatter(x, ohlc.loc[x, 'close'].values - 30, label='skitscat', color='green', s=25, marker="^")
        # predicted
        x = report.loc[report['predicted'] == 1].index.values
        plt.scatter(x, ohlc.loc[x, 'close'].values, label='skitscat', color='magenta', s=25, marker="v")
        x = report.loc[report['predicted'] == 2].index.values
        plt.scatter(x, ohlc.loc[x, 'close'].values, label='skitscat', color='cyan', s=25, marker="o")
        x = report.loc[report['predicted'] == 3].index.values
        plt.scatter(x, ohlc.loc[x, 'close'].values, label='skitscat', color='lime', s=25, marker="^")
        # Text box
        lines = [
                'Model: {}'.format(self.model),
                '==Params=='
            ] + ['{} = {}'.format(k,v) for k,v in self.params.items()] + \
            [
                '==Stats==',
                'Accuracy: {}'.format(self.accuracy()),
                'MSE: {}'.format(self.mse()),
                'Profit: {}'.format(self.profit()),
                '==Input==',
                'Symbol: {}'.format(self.symbol),
                'Dates: {} - {}'.format(report.index[0],report.index[-1])
            ]

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', alpha=0.3)

        # place a text box in upper left in axes coords
        ax.text(0.05, 0.05, '\n'.join(lines), transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)

        plt.show()