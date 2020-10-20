from sqlalchemy import Column, Integer, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from datetime import datetime
import os

Base = declarative_base()


class Job(Base):
	__tablename__ = 'jobs'
	id = Column(Integer, primary_key=True)
	dataset = Column(Text(), nullable=False)
	pipeline = Column(Text(), nullable=False)
	target = Column(Text(), nullable=False)
	dataframe = Column(Text(), nullable=True)
	status = Column(Boolean(), nullable=False)

def migrate(db_file):
	if not os.path.exists(db_file):
		engine = create_engine('sqlite:///'+db_file)
		Base.metadata.create_all(engine)
		print("Database {} created!".format(db_file))
		return True
	return False