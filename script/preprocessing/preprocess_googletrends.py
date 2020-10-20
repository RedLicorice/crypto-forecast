import pandas as pd
import os
import json
from sklearn.feature_selection import VarianceThreshold

COLUMNS = {
    'ADA' : ['ADA', 'Cardano'],
    'BCH' : ['BCH', 'Bitcoin Cash'],
    'BNB' : ['BNB', 'Binance Coin'],
    'BTC' : ['BTC', 'Bitcoin'],
    'BTG' : ['BTC', 'Bitcoin Gold'],
    'DASH' : ['DASH', 'Dash'],
    'DOGE' : ['DOGE', 'Dogecoin'],
    'EOS' : ['EOS'],
    'ETC' : ['ETC', 'Ethereum Classic'],
    'ETH' : ['ETH', 'Ethereum'],
    'IOT' : ['IOTA', 'MIOTA'],
    'LTC' : ['LTC', 'Litecoin'],
    'LINK' : ['LINK', 'Chainlink'],
    'NEO' : ['NEO'],
    'QTUM' : ['QTUM'],
    'TRX' : ['TRX', 'TRON'],
    'USDT' : ['USDT','Tether'],
    'VEN' : ['VET', 'VeChain'],
    'WAVES' : ['WAVES', 'Waves'],
    'XEM' : ['XEM', 'NEM'],
    'XMR' : ['XMR', 'Monero'],
    'XRP' : ['XRP', 'Ripple'],
    'ZEC' : ['ZEC', 'ZCash'],
    'ZRX' : ['ZRX', '0x']
}
if __name__ == '__main__':
    index = {}
    df = pd.read_csv('data/google_trends/trends_all_20101112000000_20190101000000.csv', sep=',', encoding='utf-8', index_col='date', parse_dates=True)
    for sym, columns in COLUMNS.items():
        _df = df.loc[:, columns]
        _df.columns = ['gtrends_{}_{}'.format(sym, c.lower()) for c in _df.columns]
        _df = _df.drop_duplicates().resample('D').mean().fillna(method='ffill')

        sel = VarianceThreshold()
        sel.fit(_df.values)

        sel_columns = [c for c, s in zip(_df.columns, sel.get_support()) if s]
        _df = _df.loc[:, sel_columns]
        print("{}: {} Features, {} Selected".format(sym, len(columns), len(sel_columns)))


        os.makedirs('data/preprocessed/google_trends/csv/', exist_ok=True)
        os.makedirs('data/preprocessed/google_trends/excel/', exist_ok=True)
        csv_path = 'data/preprocessed/google_trends/csv/{}.csv'.format(sym.lower())
        xls_path = 'data/preprocessed/google_trends/excel/{}.xlsx'.format(sym.lower())
        _df.to_csv(csv_path, sep=',', encoding='utf-8', index=True, index_label='Date')
        _df.to_excel(xls_path, index=True, index_label='Date')
        index[sym] = {'csv':csv_path, 'xls':xls_path}
        print('Saved {} in data/preprocessed/google_trends/'.format(sym))

    with open('data/preprocessed/google_trends/index.json', 'w') as f:
        json.dump(index, f, sort_keys=True, indent=4)