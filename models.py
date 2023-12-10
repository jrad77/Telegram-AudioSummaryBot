from sqlalchemy import create_engine, Column, Integer, String, Text, Date, DateTime, func
from sqlalchemy.ext.declarative import declarative_base

# Define the base class
Base = declarative_base()

# Define the YouTubeAudioData class
class YouTubeAudioData(Base):
    __tablename__ = 'youtube_audio_data'
    id = Column(Integer, primary_key=True)
    url = Column(String, nullable=False)
    title = Column(String)
    description = Column(Text)
    audio_file_path = Column(String)
    publish_date = Column(Date)
    index_date = Column(DateTime, default=func.now())
    transcript_file_path = Column(Text)
    speakers_count = Column(Integer)
    transcript_summary_path = Column(Text)
    summary_model_used = Column(Text)
    status = Column(String)
    updated_at = Column(DateTime, onupdate=func.now(), default=func.now())

    # non-database members
    def load_summary(self):
        with open(self.transcript_summary_path, 'r') as f:
            self.summary = f.read()

