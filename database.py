from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base  # import the Base from models.py
import config  # import configuration settings

engine = create_engine(config.DATABASE_URI)

Session = sessionmaker(bind=engine)

def init_db():
    # This will create the tables if they don't exist
    Base.metadata.create_all(engine)
