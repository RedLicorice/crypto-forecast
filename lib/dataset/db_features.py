from sqlalchemy import Column, Integer, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from datetime import datetime
import pandas as pd
import os

Base = declarative_base()

class Feature(Base):
    __tablename__ = 'features'
    id = Column(Integer, primary_key=True)
    group = Column(Text(), nullable=False)  # Feature group for this series
    day = Column(DateTime(), nullable=False)  # Day these symbols correspond to
    symbol = Column(Text(), nullable=False)  # Symbol ticker
    name = Column(Text(), nullable=False)  # Feature name
    value = Column(Float(), nullable=True) # Feature value (We use continuous features so it should be OK)

class Target(Base):
    __tablename__ = 'targets'
    id = Column(Integer, primary_key=True)
    day = Column(DateTime(), nullable=False)  # Day these symbols correspond to
    symbol = Column(Text(), nullable=False)  # Symbol ticker
    name = Column(Text(), nullable=False)  # Feature group for this series
    value = Column(Float(), nullable=True) # Feature value (We use continuous features so it should be OK)

def migrate(db_file):
    if not os.path.exists(db_file):
        engine = create_engine('sqlite:///'+db_file)
        Base.metadata.create_all(engine)
        print("Database {} created!".format(db_file))
        return True
    return False

def connect(db_file):
    engine = create_engine('sqlite:///' + db_file)
    Base.metadata.bind = engine
    session_factory = sessionmaker(bind=engine)
    DBSession = scoped_session(session_factory)
    # Create status database and tables (only if db is new)
    migrate(db_file)
    return DBSession

def add_feature_df(session, group, symbol, df: pd.DataFrame):
    for c in df.columns:
        add_feature(session, group, symbol, c, df[c])

def add_feature(session, group, symbol, name, series: pd.Series):
    for idx, value in series.iteritems():
        res = Feature(
            group=group,
            symbol=symbol,
            name=name,
            day=idx, # Ignore the warning, index must be a DateTimeIndex
            value=value
        )
        session.add(res)

def add_target(session, symbol, name, series: pd.Series):
    for idx, value in series.iteritems():
        res = Target(
            symbol=symbol,
            day=idx, # Ignore the warning, index must be a DateTimeIndex
            name=name,
            value=value
        )
        session.add(res)

def get_features(session, group, symbol):
    query = session.query(Feature).filter(
        Feature.group == group,
        Feature.symbol == symbol
    )
    df = pd.read_sql(sql=query.statement, con=query.session.bind, parse_dates=['day'])
    df = df.pivot(index='day', columns='name', values='value')
    return df

if __name__ == '__main__':
    DBSession = connect('test_features')
    session = DBSession()
    f = get_features(session, 'dbfeaturetest', 'ZRX')
