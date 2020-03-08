import pandas as pd
from functools import reduce
from lib.utils import check_duplicates

symbols = {
    'ADA': ['data/coinmetrics.io/ada.csv'],
    'BCH': ['data/coinmetrics.io/bch.csv'],
    'BNB': ['data/coinmetrics.io/bnb.csv', 'data/coinmetrics.io/bnb_mainnet.csv'],
    'BTC': ['data/coinmetrics.io/btc.csv'],
    'BTG': ['data/coinmetrics.io/btg.csv'],
    'DASH': ['data/coinmetrics.io/dash.csv'],
    'DOGE': ['data/coinmetrics.io/doge.csv'],
    'EOS': ['data/coinmetrics.io/eos.csv', 'data/coinmetrics.io/eos_eth.csv'],
    'ETC': ['data/coinmetrics.io/etc.csv'],
    'ETH': ['data/coinmetrics.io/eth.csv'],
    'LTC': ['data/coinmetrics.io/ltc.csv'],
    'LINK': ['data/coinmetrics.io/link.csv'],
    'NEO': ['data/coinmetrics.io/neo.csv'],
    'QTUM': ['data/coinmetrics.io/qtum.csv'],
    'TRX': ['data/coinmetrics.io/qtum.csv'],
    'USDT': ['data/coinmetrics.io/usdt.csv'],
    'USDTE': ['data/coinmetrics.io/usdt_eth.csv'],
    'VEN': ['data/coinmetrics.io/vet.csv'],
    'WAVES': ['data/coinmetrics.io/waves.csv'],
    'XEM': ['data/coinmetrics.io/xem.csv'],
    'XMR': ['data/coinmetrics.io/xmr.csv'],
    'XRP': ['data/coinmetrics.io/xrp.csv'],
    'ZEC': ['data/coinmetrics.io/zec.csv'],
    'ZRX': ['data/coinmetrics.io/zrx.csv'],
}

temp = []
result = None
for sym, csv in symbols.items():
    dfs = [pd.read_csv(c, sep=',', encoding='utf-8', index_col='date', parse_dates=True) for c in csv]
    df = reduce(lambda x, y: x.append(y, sort=True) if y is not None else x, dfs)

    #volume = pd.read_csv(vol, sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
    #vol_columns = [c for c in volume.columns if c.endswith('_Volume')]

    if result is None:
        result = pd.DataFrame(index=df.index)

    for c in df.columns:
        result[sym + '_' + c] = df[c]

check_duplicates(result, print=True)
result.to_csv('data/result/blockchains.csv', sep=',', encoding='utf-8', index=True, index_label='Date')
period = result.to_period(freq='W')
period.to_csv('data/result/blockchains-weekly.csv', sep=',', encoding='utf-8', index=True, index_label='Date')