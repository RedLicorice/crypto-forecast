from sqlalchemy import Column, Integer, DateTime, Text, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from datetime import datetime
from enum import IntEnum
import os

FIXED_TRADING_FEE = 0.05
# Signal labels:
# Labels and predictions use integer encoding, which is mapped as follows
class Signal(IntEnum):
    SELL, HOLD, BUY = range(3)
# Order status:
# - PENDING if the order has not been executed (for example because it is a SHORT type order)
# - OPEN if the order is open (coins have been purchased)
# - CLOSED if the order is closed (coins have been sold)
class OrderStatus(IntEnum):
    PENDING, OPEN, CLOSED = range(3)
# Order types:
# - LONG if we plan an order expecting the price to rise the next day
# - SHORT if we plan an order expecting the price to decrease the next day (so we plan to buy the next day)
class OrderType(IntEnum):
    LONG, SHORT, SPOT = range(3)

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
        return self.fiat + self.margin_fiat + self.coins*price + self.margin_coins*price

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
    type = Column(Integer(), nullable=False, default=0) # Order type ie SHORT, LONG, ...
    status = Column(Integer(), nullable=False, default=0) # Order status ie PENDING, OPEN, CLOSED

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

def migrate(db_file):
    if not os.path.exists(db_file):
        engine = create_engine('sqlite:///'+db_file)
        Base.metadata.create_all(engine)
        print("Database {} created!".format(db_file))
        return True
    return False