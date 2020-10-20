from lib.dataset import load_symbol
from lib.log import logger
from lib.trading import migrate, Base, Position, Asset
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

DBSession = None
SELL, HOLD, BUY = range(3)
def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

def trading_period(pipeline_name, parameter_dir, dataset, symbols, begin, end, window_size=30):
    begin = datetime.strptime(begin, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
    results = []
    for day in daterange(begin, end):
        signals = trailing_window_day(
            pipeline_name=pipeline_name,
            parameters=parameter_dir,
            dataset=dataset,
            symbols=symbols,
            day=day,
            window_size=window_size
        )
        equities = trading_day(day, symbols, signals)
        if equities is not None:
            results.append(equities)
    result = pd.concat(results)
    return result

def trading_day(day, symbols, signals):
    if not signals.shape[0] and not signals.shape[1]:
        return
    session = DBSession()
    result = pd.DataFrame()
    for s in symbols:
        # If there's no signal for this coin, close the position
        if not s in signals.columns:
            continue
        signal = signals['{}'.format(s)].iloc[0]
        close = signals['{}_close'.format(s)].iloc[0]
        label = signals['{}_label'.format(s)].iloc[0]
        if np.isnan(signal): # If signal is nan
            continue
        # Grab balance for current symbol
        asset = session.query(Asset).filter(Asset.symbol == s).first()
        if not asset:
            asset = Asset(
                symbol=s,
                balance=0,
                fiat = 10000
            )
            session.add(asset)
        # Handle existing positions
        positions = session.query(Position).filter(and_(Position.symbol == s, Position.closed == False, Position.open_at < datetime.utcnow())).all()
        for p in positions:
            # If close meets stop loss, close position
            if p.should_stop(close):
                logger.info('[Day: {}] Closing position {} on {} due to stop loss'.format(day, p.id, p.symbol))
                p.close(close)
                asset.remove_position(p)
                continue
            # If price increased more than 1%
            if p.price_change(close) > 0.01:
                if signal == BUY: # If signal is BUY increase stop loss
                    logger.info('[Day: {}] Increasing position {} on {}\'s stop loss by 1%'.format(day, p.id, p.symbol))
                    p.adjust_stop(0.01)
                elif signal == SELL: # If signal is SELL, take profit
                    logger.info('[Day: {}] Closing position {} on {} in profit due to SELL signal'.format(day, p.id, p.symbol))
                    p.close(close)
                    asset.remove_position(p)
                continue
            # If position is older than 3 days
            if p.get_age_in_days() >= 3:
                logger.info('[Day: {}] Closing position {} on {} due to age'.format(day, p.id, p.symbol))
                p.close(close)
                asset.remove_position(p)
                continue
            # If signal is SELL
            if signal == SELL:
                logger.info('[Day: {}] Closing position {} on {} due to SELL signal'.format(day, p.id, p.symbol))
                p.close(close)
                asset.remove_position(p)
                continue
        # Open new positions
        equity = asset.get_equity(close)
        position_price = equity * 0.1 # Each position is 10% of equity - fixed fractional
        position_coins = round(position_price / close, 8)
        if signal == BUY and asset.fiat > position_price:
            logger.info('[Day: {}] Opening position on {} due to BUY signal [Close {}, Price {}, Coins {}]'.format(day, s, close, position_price, position_coins))
            p = Position(symbol=s)
            p.open(close, position_coins, stop_loss=0.02) # Open a position for this coin at market price, with 2% stop loss
            asset.add_position(p)
            session.add(p)
        # Add result to dataframe
        result.loc[day, s] = asset.get_equity(close)
    session.commit()
    return result

def expanding_window(pipeline_name, parameters, dataset, symbol, initial_training_window=30, initial_testing_window=7):
    if isinstance(parameters, str):
        with open(parameters, 'r') as f:
            parameters = json.load(f)
    pipeline = importlib.import_module('pipelines.' + pipeline_name)
    features, target, close  = load_symbol(dataset, symbol, target='class')

    # Drop columns whose values are all NaN, as well as rows with ANY nan value, then
    # replace infinity values with nan so that they can later be imputed to a finite value
    features = features.dropna(axis='columns', how='all').replace([np.inf, -np.inf], np.nan)

    X_train = features.iloc[0:initial_training_window].copy() # copy because we're adding rows to this dataset
    X_test = features.iloc[initial_training_window:initial_training_window+initial_testing_window]
    X_trade = features.iloc[initial_training_window+initial_testing_window:]
    y_train = target.iloc[0:initial_training_window].copy() # copy because we're adding rows to this series
    y_test = target.iloc[initial_training_window:initial_training_window+initial_testing_window]
    y_trade = target.iloc[initial_training_window+initial_testing_window:]
    close_train = close.iloc[0:initial_training_window]
    close_test = close.iloc[initial_training_window:initial_training_window+initial_testing_window]
    close_trade = close.iloc[initial_training_window+initial_testing_window:]

    est = pipeline.estimator
    est.set_params(**parameters)
    logger.info('Initial training of {} estimator on {} records'.format(pipeline_name, X_train.shape[0]))
    est.fit(X_train.values, y_train.values)
    logger.info('Testing {} estimator on {} records'.format(pipeline_name, X_test.shape[0]))
    test_predictions = pd.Series(index=X_test.index, dtype='int')
    for i, idx in enumerate(X_test.index):
        sample = X_test.loc[idx]
        shaped = np.reshape(sample.values, (1, -1))
        pred = est.predict(shaped)
        test_predictions.loc[idx] = pred[0]
        X_train.loc[idx] = X_test.loc[idx]
        y_train.loc[idx] = y_test.loc[idx]
        try:
            new_est = est.fit(X_train, y_train)
            est = new_est
        except:
            # Fit failed
            logger.exception('Fit failed at index {}!'.format(idx))
            pass
        _precision = precision_score(y_test.values[0:i], test_predictions.values[0:i], average='micro', zero_division=1)
        logger.info('Estimator precision at day {} ({}): {}'.format(initial_training_window + i + 1, idx, _precision))
    logger.info('Validating {} estimator on {} records'.format(pipeline_name, X_trade.shape[0]))
    validate_predictions = pd.Series(index=X_trade.index)
    for i, idx in enumerate(X_trade.index):
        sample = X_trade.loc[idx]
        shaped = np.reshape(sample.values, (1, -1))
        pred = est.predict(shaped)
        validate_predictions.loc[idx] = pred[0]
        X_train.loc[idx] = X_trade.loc[idx]
        y_train.loc[idx] = y_trade.loc[idx]
        try:
            new_est = est.fit(X_train, y_train)
            est = new_est
        except:
            # Fit failed
            logger.exception('Fit failed at index {}!'.format(idx))
            pass
        _precision = precision_score(y_trade.values[0:i], validate_predictions.values[0:i], average='micro', zero_division=1)
        logger.info('Estimator precision at day {} ({}): {}'.format(initial_training_window + initial_testing_window + i + 1, idx, _precision))
    return est, (y_test, test_predictions, close_test), (y_trade, validate_predictions, close_trade)

def trailing_window(pipeline_name, parameters, dataset, symbol, window_size=30):
    if isinstance(parameters, str):
        with open(parameters, 'r') as f:
            parameters = json.load(f)
    pipeline = importlib.import_module('pipelines.' + pipeline_name)
    features, target, close  = load_symbol(dataset, symbol, target='class')

    # Drop columns whose values are all NaN, as well as rows with ANY nan value, then
    # replace infinity values with nan so that they can later be imputed to a finite value
    features = features.dropna(axis='columns', how='all').replace([np.inf, -np.inf], np.nan)

    est = pipeline.estimator
    est.set_params(**parameters)
    logger.info('Testing {} estimator on {} records'.format(pipeline_name, features.shape[0] - window_size))
    test_predictions = pd.Series(index=features.iloc[window_size:].index, dtype='int')
    indices = []
    # Dalla prima posizione, prendo W record alla volta
    for i in range(0, features.shape[0] - window_size, 1):
        X_train = features.iloc[i:i+window_size]
        y_train = target.iloc[i:i+window_size]
        X_test = features.iloc[i + window_size:i+window_size+1]
        y_test = target.iloc[i + window_size:i+window_size+1]
        try:
            est = est.fit(X_train, y_train)
        except:
            # Fit failed
            logger.exception('Fit failed at index {}!'.format(i))
            pass
        shaped = np.reshape(X_test.values, (1, -1))
        pred = est.predict(shaped)
        idx = X_test.index[0]
        test_predictions.loc[idx] = pred[0]
        indices.append(idx)
        _precision = precision_score(target[indices].values, test_predictions[indices].values, average='micro', zero_division=1)
        logger.info('Estimator precision at day {} ({}): {}'.format(window_size + i + 1, idx, _precision))
    close_test = close.loc[test_predictions.index]
    labels = target.loc[test_predictions.index]
    return labels, test_predictions, close_test

def trailing_window_day(pipeline_name, parameters, dataset, symbols, day, window_size=30):
    pipeline = importlib.import_module('pipelines.' + pipeline_name)
    result = pd.DataFrame()
    for symbol in symbols:
        if isinstance(parameters, str):
            if os.path.isdir(parameters):
                parameters = '{}/{}_parameters.json'.format(parameters, symbol)
            with open(parameters, 'r') as f:
                parameters = json.load(f)
        features, target, close  = load_symbol(dataset, symbol, target='class')

        # Initialize result cells to nan
        result.loc[day, '{}'.format(symbol)] = np.nan
        result.loc[day, '{}_label'.format(symbol)] = np.nan
        result.loc[day, '{}_close'.format(symbol)] = np.nan

        # Drop columns whose values are all NaN, as well as rows with ANY nan value, then
        # replace infinity values with nan so that they can later be imputed to a finite value
        features = features.dropna(axis='columns', how='all').replace([np.inf, -np.inf], np.nan)

        est = pipeline.estimator
        est.set_params(**parameters)
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

        try:
            est = est.fit(X_train, y_train)
        except:
            # Fit failed
            logger.exception('Fit failed for symbol {} at index {} (Day {})!'.format(symbol, day, i))
            pass
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
    beg = '2018-06-01'
    end = '2018-09-01'
    _symbols = [
        'ADA',
        'BCH',
        'BNB',
        'BTC',
        'BTG',
        'DASH',
        'DOGE',
        'EOS',
        'ETC',
        'ETH',
        'LTC',
        'LINK',
        'NEO',
        'QTUM',
        'TRX',
        'USDT',
        'VEN',
        'WAVES',
        'XEM',
        'XMR',
        'XRP',
        'ZEC',
        'ZRX'
    ]
    result = trading_period(
        pipeline_name='debug_xgboost',
        parameter_dir='./results/timedspline_safe_debug_xgboost_splines_experiment_171020_040945/',
        dataset='timedspline_safe',
        symbols=_symbols,
        begin=beg,
        end=end,
        window_size=150
    )
    equity_file = './equities/{}/equities.csv'.format(args.name)
    result.to_csv(equity_file, encoding='utf-8', index_label='Date')
    equity_plot = './equities/{}/equities.png'.format(args.name)
    result.plot(kind='line', title='Equity lines from {} to {}'.format(beg,end), figsize=(20,10))
    plt.savefig(equity_plot)