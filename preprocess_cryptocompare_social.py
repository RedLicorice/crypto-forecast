import pandas as pd
import json
from sklearn.feature_selection import VarianceThreshold

symbols = {
    'ADA': 'data/cryptocompare/social/social_ADA.csv',
    'BCH': 'data/cryptocompare/social/social_BCH.csv',
    'BNB': 'data/cryptocompare/social/social_BNB.csv',
    'BTC': 'data/cryptocompare/social/social_BTC.csv',
    'BTG': 'data/cryptocompare/social/social_BTG.csv',
    'DASH': 'data/cryptocompare/social/social_DASH.csv',
    'DOGE': 'data/cryptocompare/social/social_DOGE.csv',
    'EOS': 'data/cryptocompare/social/social_EOS.csv',
    'ETC': 'data/cryptocompare/social/social_ETC.csv',
    'ETH': 'data/cryptocompare/social/social_ETH.csv',
    'IOT': 'data/cryptocompare/social/social_MIOTA.csv',
    'LTC': 'data/cryptocompare/social/social_LTC.csv',
    'LINK': 'data/cryptocompare/social/social_LINK.csv',
    'NEO': 'data/cryptocompare/social/social_NEO.csv',
    'QTUM': 'data/cryptocompare/social/social_QTUM.csv',
    'TRX': 'data/cryptocompare/social/social_TRX.csv',
    'USDT': 'data/cryptocompare/social/social_USDT.csv',
    'VEN': 'data/cryptocompare/social/social_VET.csv',
    'WAVES': 'data/cryptocompare/social/social_WAVES.csv',
    'XEM': 'data/cryptocompare/social/social_XEM.csv',
    'XMR': 'data/cryptocompare/social/social_XMR.csv',
    'XRP': 'data/cryptocompare/social/social_XRP.csv',
    'ZEC': 'data/cryptocompare/social/social_ZEC.csv',
    'ZRX': 'data/cryptocompare/social/social_ZRX.csv'
}
if __name__ == '__main__':
    index = {}
    for sym, csv in symbols.items():
        df = pd.read_csv(csv, sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        # These datasets have zero-filled rows, we need to clear them
        df = df[(df.select_dtypes(include=['number']) != 0).any(1)]
        df = df.dropna(thresh=30, axis='columns').dropna(thresh=5)
        # Remove zero-variance features
        sel = VarianceThreshold()
        sel.fit_transform(df)
        sup = sel.get_support()

        df = df[[name for flag, name in zip(sup, df.columns) if flag]]

        csv_path = 'data/preprocessed/cryptocompare_social/csv/{}.csv'.format(sym.lower())
        xls_path = 'data/preprocessed/cryptocompare_social/excel/{}.xlsx'.format(sym.lower())
        df.to_csv(csv_path, sep=',', encoding='utf-8', index=True, index_label='Date')
        df.to_excel(xls_path, index=True, index_label='Date')
        index[sym] = {'csv':csv_path, 'xls':xls_path}
        print('Saved {} in data/preprocessed/cryptocompare_social/'.format(sym))

    with open('data/preprocessed/cryptocompare_social/index.json', 'w') as f:
        json.dump(index, f, sort_keys=True, indent=4)