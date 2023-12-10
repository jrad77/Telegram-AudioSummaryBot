from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.sql import text
import os
SQLALCHEMY_DATABASE_URI = os.getenv('SQLALCHEMY_DATABASE_URI')


engine = create_engine(SQLALCHEMY_DATABASE_URI) 
metadata = MetaData()

# Reflect the table
users = Table('youtube_audio_data', metadata, autoload_with=engine)

# Add the new column
with engine.connect() as connection:
   connection.execute(text("ALTER TABLE youtube_audio_data ADD COLUMN summary_model_used TEXT"))