from lib.dataset import load_symbol
from lib.trading.models import migrate, Base, OrderType, Signal
from lib.trading.exchange import Exchange
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker, scoped_session
from lib.log import logger
import logging
from sklearn.metrics import precision_score
import json, os, argparse, importlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from scipy.interpolate import UnivariateSpline
import pickle

DBSession = None


def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

def trading_period(pipeline_name, parameter_dir, dataset, symbols, begin, end, window_size=30, order_size=0.1, cache_dir=None):
    begin = datetime.strptime(begin, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
    results = []
    signal_history = pd.DataFrame()
    for day in daterange(begin, end):
        # Re train the model
        day_signals = trailing_window_day(
            pipeline_name=pipeline_name,
            parameters=parameter_dir,
            dataset=dataset,
            symbols=symbols,
            day=day,
            window_size=window_size,
            cache_dir=cache_dir
        )
        equities = trading_day(day, symbols, day_signals, order_size=order_size, history=signal_history)
        signal_history = pd.concat([signal_history, day_signals], axis=0)
        results.append(equities)
    result = pd.concat(results)
    return result

def check_signal(signal, close, last_close):
    if not last_close:
        return True
    labels = ["SELL","HOLD","BUY"]
    last_percent = (close - last_close) / last_close
    if last_percent >= 0.01 and signal != Signal.BUY:
        raise Exception("Wrong signal, expected BUY got {} PCT: {}".format(labels[int(signal)], last_percent))
    elif last_percent <= -0.01 and signal != Signal.SELL:
        raise Exception("Wrong signal, expected SELL got {} PCT: {}".format(labels[int(signal)], last_percent))
    elif last_percent >= -0.01 and last_percent < 0.01 and signal != Signal.HOLD:
        raise Exception("Wrong signal, expected HOLD got {} PCT: {}".format(labels[int(signal)], last_percent))

def trading_day(day, symbols, signals, order_size=0.1, history=None):
    if not signals.shape[0] and not signals.shape[1]:
        return
    session = DBSession()
    result = pd.DataFrame()
    exchange = Exchange(session)
    for s in symbols:
        # If there's no signal for this coin, close the position
        if not s in signals.columns:
            continue
        signal = signals['{}'.format(s)].iloc[0]
        close = signals['{}_close'.format(s)].iloc[0]
        label = signals['{}_label'.format(s)].iloc[0]

        if np.isnan(signal): # If signal is nan
            continue
        if history is not None and not history.empty:
            signal_history = history['{}'.format(s)]
            close_history = history['{}_close'.format(s)]
            label_history = history['{}_label'.format(s)]
            precision = precision_score(label_history.values, signal_history.values, average='micro', zero_division=True)
            # Fit an spline on available historical data, needs at least 7 days of activity
            history_length = close_history.shape[0]
            if history_length > 0:
                # Check last label is correct
                check_signal(label_history.values[-1], close, close_history.values[-1])
                #--
                # hist = close_history.copy()
                # hist.loc[day] = close
                # pct = hist.pct_change().values[-1]
                # if history_length >= 7:
                #     x_space = np.linspace(0, history_length - 1, history_length)
                #     close_spline = UnivariateSpline(x_space, close_history.values, s=0, k=4)
                #     d1 = close_spline(history_length - 1, nu=1)
                #     d2 = close_spline(history_length - 1, nu=2)
                #     logger.info(
                #         "[Trading day: {}] {} | Signal: {} True: {} Precision: {} | Close: {} Pct: {} d1: {} d2: {}".format(
                #             day, s, signal, label, precision, close, pct, d1, d2
                #         ))
                # else:
                #     logger.info("[Trading day: {}] {} | Signal: {} True: {} Precision: {} | Close: {} Pct: {}".format(
                #         day, s, signal, label, precision, close, pct
                #     ))
        #signal = label
        # Grab balance for current symbol
        asset = exchange.get_or_create_asset(s, margin_fiat=10000, coins=0)
        #
        # Order management
        #
        # Manage LONG orders
        open_longs = exchange.get_open_long(asset)
        for o in open_longs:
            # If close meets stop loss, close position
            if o.should_stop(close):
                logger.info('[Day: {}] Closing long position {} on {} due to stop loss'.format(day, o.id, o.symbol))
                exchange.close_order(day, asset, o, close)
                continue

            # If signal is SELL or position has a 1% profit
            if signal == Signal.SELL:
                logger.info('[Day: {}] Closing long position {} on {} due to SELL signal'.format(day, o.id, o.symbol))
                exchange.close_order(day, asset, o, close)
                continue

        # Manage SHORT orders
        open_shorts = exchange.get_open_short(asset)
        for o in open_shorts:
            # If close meets stop loss, close position
            if o.should_stop(close):
                logger.info('[Day: {}] Closing short position {} on {} due to stop loss'.format(day, o.id, o.symbol))
                exchange.close_order(day, asset, o, close)
                continue
            # If signal is BUY we're going to lose money, so we close position
            if signal == Signal.BUY:
                logger.info('[Day: {}] Closing short position {} on {} due to BUY signal'.format(day, o.id, o.symbol))
                exchange.close_order(day, asset, o, close)
                continue
            # If signal is HOLD and position is old
            if o.get_age_in_days(day) > 2 and signal == Signal.HOLD:
                logger.info('[Day: {}] Closing short position {} on {} due to age'.format(day, o.id, o.symbol))
                exchange.close_order(day, asset, o, close)
                continue

        #
        # Open new positions
        #
        # Determine position sizing
        position_coins = asset.position_size(close, order_size)
        # Open the order
        if signal == Signal.BUY:
            logger.info('[Day: {}] Opening long position on {} due to BUY signal [Close {}, Price {}, Coins {}]'.format(day, s, close, position_coins*close, position_coins))
            o = exchange.open_order(day, OrderType.LONG, asset, position_coins, close, stop_loss=-0.01) # Stop loss is -1%
            if not o:
                logger.error("LONG FAILED")
        elif signal == Signal.SELL:
            logger.info('[Day: {}] Opening short position on {} due to BUY signal [Close {}, Price {}, Coins {}]'.format(day, s, close, position_coins*close, position_coins))
            o = exchange.open_order(day, OrderType.SHORT, asset, position_coins, close, stop_loss=0.01) # Stop loss is +1%
            if not o:
                logger.error("SHORT FAILED")
        # Add result to dataframe
        result.loc[day, s] = asset.equity(close)
    session.commit()
    return result


def trailing_window_day(pipeline_name, parameters, dataset, symbols, day, window_size=30, cache_dir=None):
    pipeline = importlib.import_module('pipelines.' + pipeline_name)
    result = pd.DataFrame()
    for symbol in symbols:
        if isinstance(parameters, str):
            if os.path.isdir(parameters):
                parameters = '{}/{}_parameters.json'.format(parameters, symbol)
            with open(parameters, 'r') as f:
                parameters = json.load(f)
        #elif isinstance(parameters, int):
            #Todo: perform parameter selection using data before the window if available?
        features, target, close  = load_symbol(dataset, symbol, target='class')

        # Initialize result cells to nan
        result.loc[day, '{}'.format(symbol)] = np.nan
        result.loc[day, '{}_label'.format(symbol)] = np.nan
        result.loc[day, '{}_close'.format(symbol)] = np.nan

        # Drop columns whose values are all NaN, as well as rows with ANY nan value, then
        # replace infinity values with nan so that they can later be imputed to a finite value
        features = features.dropna(axis='columns', how='all').replace([np.inf, -np.inf], np.nan)

        #logger.info('Testing {} estimator on {} records'.format(pipeline_name, features.shape[0] - window_size))
        #test_predictions = pd.Series(index=features.iloc[window_size:].index, dtype='int')
        if day not in features.index:
            logger.error('No data on symbol {} for day {}!'.format(symbol, day))
            continue
        i = features.index.get_loc(day)
        if i <= window_size:
            logger.error('Not enough data on symbol {} to train model for day {}!'.format(symbol, day))
            continue
        X_train = features.iloc[i - window_size:i]
        y_train = target.iloc[i - window_size:i]
        X_test = features.iloc[i:i+1]
        y_test = target.iloc[i:i+1]
        close_test = close.iloc[i:i+1]

        est = None
        cached_path = None
        if cache_dir is not None and os.path.isdir(cache_dir):
            cached_name = "model_{}_{}_{}_{}.p".format(pipeline_name, dataset, symbol, day.strftime("D%Y%m%dT%H%M%S"))
            cached_path ="{}/{}".format(cache_dir, cached_name)
            if os.path.exists(cached_path):
                with open(cached_path, "rb") as f:
                    est = pickle.load(f)
        if est is None:
            est = pipeline.estimator
            est.set_params(**parameters)
            try:
                est = est.fit(X_train, y_train)
            except:
                # Fit failed
                logger.exception('Fit failed for symbol {} at index {} (Day {})!'.format(symbol, day, i))
                pass
            if cached_path:
                with open(cached_path, "wb") as f:
                    pickle.dump(est, f)
        shaped = np.reshape(X_test.values, (1, -1))
        pred = est.predict(shaped)
        result.loc[day, '{}'.format(symbol)] = pred[0]
        result.loc[day, '{}_label'.format(symbol)] = y_test.iloc[0]
        result.loc[day, '{}_close'.format(symbol)] = close_test.iloc[0]
    return result

if __name__ == '__main__':
    np.random.seed(0)
    parser = argparse.ArgumentParser(
        description='Build and tune models, collect results')
    parser.add_argument('-n', dest='name', nargs='?', default='equity_test',
                        help="Name for the current equity")  # nargs='?', default='all_merged.index_improved',
    args = parser.parse_args()
    os.makedirs('./equities/{}/'.format(args.name), exist_ok=True)
    models_cache_dir = './equities/{}/models'.format(args.name)
    os.makedirs(models_cache_dir, exist_ok=True)
    db_file = './equities/{}/status.db'.format(args.name)
    if os.path.exists(db_file):
        os.remove(db_file)
    engine = create_engine('sqlite:///' + db_file)
    Base.metadata.bind = engine
    session_factory = sessionmaker(bind=engine)
    DBSession = scoped_session(session_factory)
    migrate(db_file) # Create status database and tables
    logger.setup(
        filename='./equities/{}/log.txt'.format(args.name),
        filemode='w',
        root_level=logging.DEBUG,
        log_level=logging.DEBUG,
        logger='equity'
    )
    # result = trailing_window_day(pipeline_name='debug_xgboost',
    #                        parameters='./results/timedspline_safe_debug_xgboost_splines_experiment_171020_040945/',
    #                        dataset='timedspline_safe',
    #                        symbols=['ADA', 'BTC'],
    #                        day='2018-08-01',
    #                        window_size=150
    #                        )
    beg = '2017-06-01'
    end = '2017-09-01'
    _symbols = [
        # 'ADA',
        # 'BCH',
        # 'BNB',
        'BTC',
        # 'BTG',
        # 'DASH',
        # 'DOGE',
        # 'EOS',
        # 'ETC',
        # 'ETH',
        # 'LTC',
        # 'LINK',
        # 'NEO',
        # 'QTUM',
        # 'TRX',
        # 'USDT',
        # 'VEN',
        # 'WAVES',
        # 'XEM',
        # 'XMR',
        # 'XRP',
        # 'ZEC',
        # 'ZRX'
    ]
    result = trading_period(
        pipeline_name='debug_xgboost',
        parameter_dir='./results/timedspline_safe_debug_xgboost_splines_experiment_171020_040945/',
        cache_dir=models_cache_dir,
        dataset='timedspline_safe',
        symbols=_symbols,
        begin=beg,
        end=end,
        window_size=200,
        order_size=0.1
    )
    equity_csv = './equities/{}/equities.csv'.format(args.name)
    equity_xlsx = './equities/{}/equities.xlsx'.format(args.name)
    result.to_csv(equity_csv, encoding='utf-8', index_label='Date')
    result.to_excel(equity_xlsx, index=True, index_label='Date')
    equity_plot = './equities/{}/equities.png'.format(args.name)
    result.plot(kind='line', title='Equity lines from {} to {}'.format(beg,end), figsize=(20,10), grid=True)
    plt.savefig(equity_plot)
    plt.close()
    for s in _symbols:
        if s not in result.columns:
            continue
        equity_plot = './equities/{}/{}_equity.png'.format(args.name, s)
        result[s].plot(kind='line', title='{} Equity lines from {} to {}'.format(s, beg, end), figsize=(20, 10), grid=True)
        plt.savefig(equity_plot)
        plt.close()