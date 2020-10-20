from googletrends import GoogleTrendsScraper
import os

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
# 2010-11-12 to 2018-12-31
START_DATE="2010-11-12"
END_DATE="2019-01-01"

def scrape(startDt, endDt):
    for sym, terms in COLUMNS.items():
        stateFile = './state/{}-state.json'.format(sym)
        gs = GoogleTrendsScraper(basename=sym)
        if os.path.exists(stateFile) and os.stat(stateFile).st_size > 0:
            state = gs.load_state(stateFile)
        else:
            state = gs.build_state(startDt, endDt, terms)
            gs.save_state(state, stateFile)
        gs.run(state)

if __name__ == '__main__':
    scrape(START_DATE, END_DATE)