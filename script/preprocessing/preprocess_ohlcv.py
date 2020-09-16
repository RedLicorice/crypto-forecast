import pandas as pd
from functools import reduce
from old.lib.utils import check_duplicates
import json

print('Merging OHLC Candles and Volume data')
candles = [
    'data/polito/2011_candle.csv',
    'data/polito/2012_candle.csv',
    'data/polito/2013_candle.csv',
    'data/polito/2014_candle.csv',
    'data/polito/2015_candle.csv',
    'data/polito/2016_candle.csv',
    'data/polito/2017_candle.csv',
    'data/polito/2018_candle.csv'
]
volumes = [
    'data/polito/2011_volume.csv',
    'data/polito/2012_volume.csv',
    'data/polito/2013_volume.csv',
    'data/polito/2014_volume.csv',
    'data/polito/2015_volume.csv',
    'data/polito/2016_volume.csv',
    'data/polito/2017_volume.csv',
    'data/polito/2018_volume.csv'
]

temp = []
for cdl, vol in zip(candles, volumes):
    ohlc = pd.read_csv(cdl, sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
    volume = pd.read_csv(vol, sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
    vol_columns = [c for c in volume.columns if c.endswith('_Volume')]
    # Drop columns which are unnamed
    unnamed_columns = [c for c in ohlc.columns if 'Unnamed' in c]
    if unnamed_columns:
        print("Dropping columns {} from candle file {}".format(unnamed_columns, cdl))
        ohlc.drop(labels=unnamed_columns, axis='columns', inplace=True)
    for c in vol_columns:
        if 'Unnamed' in c:
            print("Dropping column {} from volume file {}".format(c, vol))
            continue
        ohlc[c] = volume[c]
    temp.append(ohlc)

whole = reduce(lambda x, y: x.append(y), temp)

if check_duplicates(whole):
    print("De-duplicating")
    whole = whole.loc[~whole.index.duplicated(keep='first')]
    check_duplicates(whole, print=True)

symbols = set([c.split('_')[0] for c in whole.columns if '_' in c])
index = {}
for s in symbols:
    df = whole[[w for w in whole.columns if w.startswith(s)]]
    df.columns = [c.replace(s+'_', '').replace(s,'close').lower() for c in df.columns]
    df = df.dropna()
    csv_path = 'data/preprocessed/ohlcv/csv/{}.csv'.format(s.lower())
    xls_path = 'data/preprocessed/ohlcv/excel/{}.xlsx'.format(s.lower())

    df.to_csv(csv_path, sep=',', encoding='utf-8', index=True, index_label='Date')
    df.to_excel(xls_path, index=True, index_label='Date')

    index[s] = {'csv':csv_path, 'xls':xls_path}

    print('Saved {} in data/preprocessed/ohlcv/'.format(s))

with open('data/preprocessed/ohlcv/index.json', 'w') as f:
    json.dump(index, f, sort_keys=True, indent=4)
#whole.to_csv('data/result/ohlcv.csv', sep=',', encoding='utf-8', index=True, index_label='Date')
print('Done')