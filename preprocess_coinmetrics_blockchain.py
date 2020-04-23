import pandas as pd
import json
from sklearn.feature_selection import VarianceThreshold

symbols = {
    'ADA': 'data/coinmetrics.io/ada.csv',
    'BCH': 'data/coinmetrics.io/bch.csv',
    'BNB': 'data/coinmetrics.io/bnb.csv',
    'BNB-MAIN': 'data/coinmetrics.io/bnb_mainnet.csv',
    'BTC': 'data/coinmetrics.io/btc.csv',
    'BTG': 'data/coinmetrics.io/btg.csv',
    'DASH': 'data/coinmetrics.io/dash.csv',
    'DOGE': 'data/coinmetrics.io/doge.csv',
    'EOS': 'data/coinmetrics.io/eos_eth.csv',
    'EOS-MAIN': 'data/coinmetrics.io/eos.csv',
    'ETC': 'data/coinmetrics.io/etc.csv',
    'ETH': 'data/coinmetrics.io/eth.csv',
    'LTC': 'data/coinmetrics.io/ltc.csv',
    'LINK': 'data/coinmetrics.io/link.csv',
    'NEO': 'data/coinmetrics.io/neo.csv',
    'QTUM': 'data/coinmetrics.io/qtum.csv',
    'TRX': 'data/coinmetrics.io/trx.csv',
    'USDT': 'data/coinmetrics.io/usdt.csv',
    'USDT-ETH': 'data/coinmetrics.io/usdt_eth.csv',
    'VEN': 'data/coinmetrics.io/vet.csv',
    'WAVES': 'data/coinmetrics.io/waves.csv',
    'XEM': 'data/coinmetrics.io/xem.csv',
    'XMR': 'data/coinmetrics.io/xmr.csv',
    'XRP': 'data/coinmetrics.io/xrp.csv',
    'ZEC': 'data/coinmetrics.io/zec.csv',
    'ZRX': 'data/coinmetrics.io/zrx.csv',
}

if __name__ == '__main__':
    index = {}
    for sym, csv in symbols.items():

        df = pd.read_csv(csv, sep=',', encoding='utf-8', index_col='date', parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        # Drop columns containing nans, keep those with at least 30 non-na values
        # Drop rows containing nans, keep those with at least 5 non-na values
        df = df.dropna(thresh=30, axis='columns').dropna(thresh=5)
        # Remove zero-variance features
        sel = VarianceThreshold()
        sel.fit_transform(df)
        sup = sel.get_support()
        df = df[[name for flag, name in zip(sup, df.columns) if flag]]

        csv_path = 'data/preprocessed/coinmetrics.io/csv/{}.csv'.format(sym.lower())
        xls_path = 'data/preprocessed/coinmetrics.io/excel/{}.xlsx'.format(sym.lower())
        df.to_csv(csv_path, sep=',', encoding='utf-8', index=True, index_label='Date')
        df.to_excel(xls_path, index=True, index_label='Date')
        index[sym] = {'csv':csv_path, 'xls':xls_path}
        print('Saved {} in data/preprocessed/coinmetrics.io/'.format(sym))

    with open('data/preprocessed/coinmetrics.io/index.json', 'w') as f:
        json.dump(index, f, sort_keys=True, indent=4)