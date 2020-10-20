import talib
import pandas as pd

def get_talib_patterns(ohlcv):
    candle_names = [
        'CDL2CROWS',
        'CDL3BLACKCROWS',
        'CDL3INSIDE',
        'CDL3LINESTRIKE',
        'CDL3OUTSIDE',
        'CDL3STARSINSOUTH',
        'CDL3WHITESOLDIERS',
        'CDLABANDONEDBABY',
        'CDLADVANCEBLOCK',
        'CDLBELTHOLD',
        'CDLBREAKAWAY',
        'CDLCLOSINGMARUBOZU',
        'CDLCONCEALBABYSWALL',
        'CDLCOUNTERATTACK',
        'CDLDARKCLOUDCOVER',
        'CDLDOJI',
        'CDLDOJISTAR',
        'CDLDRAGONFLYDOJI',
        'CDLENGULFING',
        'CDLEVENINGDOJISTAR',
        'CDLEVENINGSTAR',
        'CDLGAPSIDESIDEWHITE',
        'CDLGRAVESTONEDOJI',
        'CDLHAMMER',
        'CDLHANGINGMAN',
        'CDLHARAMI',
        'CDLHARAMICROSS',
        'CDLHIGHWAVE',
        'CDLHIKKAKE',
        'CDLHIKKAKEMOD',
        'CDLHOMINGPIGEON',
        'CDLIDENTICAL3CROWS',
        'CDLINNECK',
        'CDLINVERTEDHAMMER',
        'CDLKICKING',
        'CDLKICKINGBYLENGTH',
        'CDLLADDERBOTTOM',
        'CDLLONGLEGGEDDOJI',
        'CDLLONGLINE',
        'CDLMARUBOZU',
        'CDLMATCHINGLOW',
        'CDLMATHOLD',
        'CDLMORNINGDOJISTAR',
        'CDLMORNINGSTAR',
        'CDLONNECK',
        'CDLPIERCING',
        'CDLRICKSHAWMAN',
        'CDLRISEFALL3METHODS',
        'CDLSEPARATINGLINES',
        'CDLSHOOTINGSTAR',
        'CDLSHORTLINE',
        'CDLSPINNINGTOP',
        'CDLSTALLEDPATTERN',
        'CDLSTICKSANDWICH',
        'CDLTAKURI',
        'CDLTASUKIGAP',
        'CDLTHRUSTING',
        'CDLTRISTAR',
        #'CDLUNIQUE3RIVE',
        'CDLUPSIDEGAP2CROWS',
        'CDLXSIDEGAP3METHODS'
    ]
    df = pd.DataFrame(index=ohlcv.index)
    # create columns for each pattern
    for candle in candle_names:
        # below is same as;
        # df["CDL3LINESTRIKE"] = talib.CDL3LINESTRIKE(op, hi, lo, cl)
        df[candle] = getattr(talib, candle)(ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close)
    return df