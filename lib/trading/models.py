from sqlalchemy import Column, Integer, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from datetime import datetime
import os

FIXED_TRADING_FEE = 0.05
# Signal labels:
# Labels and predictions use integer encoding, which is mapped as follows
class SignalType:
    SELL = 'SELL'
    HOLD = 'HOLD'
    BUY = 'BUY'

    @staticmethod
    def parse(signal: int):
        if signal < 0 or signal > 2:
            raise Exception("Invalid signal!")
        res = [SignalType.SELL, SignalType.HOLD, SignalType.BUY]
        return res[signal]

# Order status:
# - PENDING if the order has not been executed (for example because it is a SHORT type order)
# - OPEN if the order is open (coins have been purchased)
# - CLOSED if the order is closed (coins have been sold)
class OrderStatus:
    OPEN = 'OPEN'
    CLOSED = 'CLOSED'
# Order types:
# - LONG if we plan an order expecting the price to rise the next day
# - SHORT if we plan an order expecting the price to decrease the next day (so we plan to buy the next day)
class OrderType:
    LONG = 'LONG'
    SHORT = 'SHORT'
    SPOT = 'SPOT'

Base = declarative_base()

class Asset(Base):
    __tablename__ = 'asset'
    id = Column(Integer, primary_key=True)
    symbol = Column(Text(), nullable=False, unique=True) # Which coin is this? BTC, ETH, BNB, ...
    coins = Column(Float(decimal_return_scale=8), nullable=False, default=0) # How many coins do we own for this asset
    fiat = Column(Float(), nullable=False, default=0) # Fiat available for this asset, used for spot orders (balance)
    margin_fiat = Column(Float(), nullable=False, default=0)
    margin_coins = Column(Float(decimal_return_scale=8), nullable=False, default=0)
    long_allowance = Column(Float(), nullable=False, default=5000) # Margin allowance for long margin trades on this asset
    short_allowance = Column(Float(decimal_return_scale=8), nullable=False, default=2.5) # Margin allowance for short margin trades on this asset
    long_orders = Column(Integer(), nullable=False, default=0) # How many open shorts
    short_orders = Column(Integer(), nullable=False, default=0) # How many open longs
    spot_orders = Column(Integer(), nullable=False, default=0) # How many open spots

    def equity(self, price):
        equity = self.fiat + self.coins*price
        # Do not consider lent balance for equity!
        # equity += self.margin_fiat + self.margin_coins*price
        return equity

    def position_size(self, price, order_size):
        equity = self.equity(price)
        # Determine position sizing
        if order_size < 1:  # If it's a float, use fixed fractional strategy
            order_price = equity * order_size  # Each position is 10% of equity - fixed fractional
        else:  # If it's an integer, use fixed amount strategy
            order_price = order_size
        position_coins = round(order_price / price, 8)
        return position_coins

class Order(Base):
    __tablename__ = 'order'

    id = Column(Integer, primary_key=True)
    symbol = Column(Text(), nullable=False) # Which coin is this? BTC, ETH, BNB, ...
    type = Column(Text(), nullable=False) # Order type ie SHORT, LONG, ...
    status = Column(Text(), nullable=False, default=0) # Order status ie PENDING, OPEN, CLOSED

    coins = Column(Float(), nullable=False, default=0.0)  # Amount of coins in order

    open_price = Column(Float(), nullable=False) # Closing price when the position was opened
    close_price = Column(Float(), nullable=True) # Closing price when the position was closed

    last_price = Column(Float(), nullable=True)  # Price when stop loss was last checked
    stop_loss = Column(Float(), nullable=True) # Stop loss

    open_at = Column(DateTime(), default=datetime.utcnow) # When the order should be executed
    closed_at = Column(DateTime(), nullable=True) # When the order should be closed

    profit = Column(Float(), nullable=True)
    open_fee = Column(Float(), nullable=True)
    close_fee = Column(Float(), nullable=True)

    def should_stop(self, close):
        if self.type == OrderType.LONG:
            return close <= self.stop_loss
        elif self.type == OrderType.SHORT:
            return close >= self.stop_loss
        elif self.type == OrderType.SPOT:
            return close <= self.stop_loss

    def get_age_in_days(self, day = None):
        if not day:
            day = datetime.utcnow()
        return (day - self.open_at).days

class OrderLog(Base):
    __tablename__ = 'order_log'
    id = Column(Integer, primary_key=True)
    symbol = Column(Text(), nullable=False) # Which coin is this? BTC, ETH, BNB, ...
    type = Column(Text(), nullable=False, default=0) # Order type ie SHORT, LONG, ...
    #order = Column(Integer(), nullable=False) # Order ID
    status = Column(Text(), nullable=False, default=0) # New order status ie PENDING, OPEN, CLOSED
    price = Column(Float(), nullable=False) # Price at which the command has to be executed, ie for a CLOSE it's close price.
    timestamp = Column(DateTime(), default=datetime.utcnow)  # When the status change happened


class Signal(Base):
    __tablename__ = 'signal'
    id = Column(Integer, primary_key=True)
    day = Column(DateTime(), nullable=False)  # Day these symbols correspond to
    symbol = Column(Text(), nullable=False) # Symbol ticker
    signal = Column(Text(), nullable=False)
    label = Column(Text(), nullable=False)
    close = Column(Float(), nullable=False)

class Equity(Base):
    __tablename__ = 'equity'
    id = Column(Integer, primary_key=True)
    day = Column(DateTime(), nullable=False)  # Day these symbols correspond to
    symbol = Column(Text(), nullable=False)  # Symbol ticker
    equity = Column(Float(), nullable=False)
    longs = Column(Integer(), nullable=False, default=0)  # How many open shorts
    shorts = Column(Integer(), nullable=False, default=0)  # How many open longs

class Feature(Base):
    __tablename__ = 'features'
    id = Column(Integer, primary_key=True)
    group = Column(Text(), nullable=False)  # Feature group for this series
    day = Column(DateTime(), nullable=False)  # Day these symbols correspond to
    symbol = Column(Text(), nullable=False)  # Symbol ticker
    name = Column(Text(), nullable=False)  # Feature name
    value = Column(Float(), nullable=False) # Feature value (We use continuous features so it should be OK)

def migrate(db_file):
    if not os.path.exists(db_file):
        engine = create_engine('sqlite:///'+db_file)
        Base.metadata.create_all(engine)
        print("Database {} created!".format(db_file))
        return True
    return False