from sqlalchemy import Column, Integer, DateTime, Text, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from datetime import datetime
import os

Base = declarative_base()

class Asset(Base):
    __tablename__ = 'asset'
    id = Column(Integer, primary_key=True)
    symbol = Column(Text(), nullable=False, unique=True) # Which coin is this? BTC, ETH, BNB, ...
    balance = Column(Float(decimal_return_scale=8), nullable=False) # How many coins do we own for this asset
    fiat = Column(Float(), nullable=False) # How many FIAT is available for this asset

    def get_equity(self, close):
        return self.balance*close + self.fiat

    def add_position(self, p):
        if self.fiat < p.coins * p.open_price:
            raise Exception("Not enough fiat to open position!")
        self.fiat -= p.coins * p.open_price
        self.balance += p.coins

    def remove_position(self, p):
        if self.balance < p.coins:
            #raise Exception("Not enough balance to close position!")
            self.balance = p.coins
        self.fiat += p.coins * p.close_price
        self.balance -= p.coins

class Position(Base):
    __tablename__ = 'position'
    id = Column(Integer, primary_key=True)
    symbol = Column(Text(), nullable=False) # Which coin is this? BTC, ETH, BNB, ...
    open_price = Column(Float(), nullable=False) # Closing price when the position was opened
    close_price = Column(Float(), nullable=True) # Closing price when the position was closed
    stop_loss = Column(Float(), nullable=True) # Stop loss (If close < stop_loss, position gets closed)
    coins = Column(Float(), nullable=False) # Amount of coins purchased
    closed = Column(Boolean(), default=False) # Open/Closed
    open_at = Column(DateTime(), default=datetime.utcnow) # Open/Closed
    closed_at = Column(DateTime(), nullable=True) # Open/Closed

    def get_earning(self):
        return self.close_price*self.coins - self.open_price*self.coins

    def open(self, price, coins, stop_loss=None):
        self.open_price = price
        self.coins = coins
        self.stop_loss = price - ((price * stop_loss) /100) if stop_loss else None
        self.close_price = None
        self.closed = False
        self.open_at = datetime.utcnow()
        self.closed_at = None

    def close(self, price):
        self.close_price = price
        self.closed = True
        self.closed_at = datetime.utcnow()

    def adjust_stop(self, pct):
        self.stop_loss += self.stop_loss * pct

    def should_stop(self, close):
        return close <= self.stop_loss

    def price_change(self, price=None):
        if not price and self.close_price:
            price = self.close_price
        return ((price - self.open_price) / self.open_price)

    def get_age_in_days(self):
        return (datetime.utcnow() - self.open_at).days

def migrate(db_file):
    if not os.path.exists(db_file):
        engine = create_engine('sqlite:///'+db_file)
        Base.metadata.create_all(engine)
        print("Database {} created!".format(db_file))
        return True
    return False