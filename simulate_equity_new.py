from lib.log import logger
from lib.trading.exchange import Exchange
from lib.trading.actor import Actor
from lib.trading.ml_strategy import MLStrategy
from lib.trading.backtesting import Backtest
from lib.trading.models import migrate, Base, OrderType, SignalType
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import numpy as np
import argparse, os, logging

BEGIN = '2018-09-01'
END = '2018-09-30'
SYMBOLS = [
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
    # 'USDT',
    'VEN',
    'WAVES',
    'XEM',
    'XMR',
    'XRP',
    'ZEC',
    'ZRX'
]

if __name__ == '__main__':
    np.random.seed(0)
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Build and tune models, collect results')
    parser.add_argument('-n', dest='name', nargs='?', default='equity_test',
                        help="Name for the current equity")  # nargs='?', default='all_merged.index_improved',
    args = parser.parse_args()
    # Create status directory and DB file
    base_dir = './equities/{}/'.format(args.name)
    os.makedirs(base_dir, exist_ok=True)
    db_file = '{}/status.db'.format(base_dir)
    # if os.path.exists(db_file):
    #     os.remove(db_file)
    # Setup logging
    logger.setup(
        filename='{}/log.txt'.format(base_dir),
        filemode='w',
        root_level=logging.DEBUG,
        log_level=logging.DEBUG,
        logger='equity'
    )
    # Create SQLAlchemy engine and bind the models
    engine = create_engine('sqlite:///' + db_file)
    Base.metadata.bind = engine
    session_factory = sessionmaker(bind=engine)
    DBSession = scoped_session(session_factory)
    # Create status database and tables (only if db is new)
    migrate(db_file)
    # Create exchange instance
    exchange = Exchange(DBSession)
    strategies = {}
    for s in SYMBOLS:
        # Deposit initial asset amount
        session = DBSession()
        asset = exchange.get_or_create_asset(s, fiat=10000)
        session.commit()
        # Create strategy instance for symbol
        strategies[s] = MLStrategy(
            pipeline='debug_xgboost',
            parameters='./results/timedspline_debug_xgboost_splines_experiment_171220_060755/{}_parameters.json'.format(s),
            dataset='timedspline',
            symbol=s,
            target='class',
            cache_base_dir=base_dir
        )
    # Create actor instance
    actor = Actor()
    simulator = Backtest(
        name='test',
        exchange=exchange,
        strategies=strategies,
        sessionFactory=DBSession,
        actor=actor
    )
    # simulator.trading_period(BEGIN, END)
    for s in SYMBOLS:
        simulator.plot_period(s, BEGIN, END, "{}/{}_trading.png".format(base_dir, s))

    # Plot