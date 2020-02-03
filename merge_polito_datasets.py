import pandas as pd
from functools import reduce
from lib.utils import check_duplicates

print('Merging')
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
    for c in vol_columns:
        ohlc[c] = volume[c]
    temp.append(ohlc)

whole = reduce(lambda x, y: x.append(y), temp)

if check_duplicates(whole):
    print("De-duplicating")
    whole = whole.loc[~whole.index.duplicated(keep='first')]
    check_duplicates(whole, print=True)

whole.to_csv('data/result/polito.csv', sep=',', encoding='utf-8', index=True, index_label='Date')
period = whole.to_period(freq='W')
period.to_csv('data/result/polito-weekly.csv', sep=',', encoding='utf-8', index=True, index_label='Date')