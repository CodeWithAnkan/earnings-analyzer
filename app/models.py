from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from app.database import Base

class Transcript(Base):
    __tablename__ = "transcripts"

    id          = Column(Integer, primary_key=True, index=True)
    ticker      = Column(String(10), index=True)
    company     = Column(String(100))
    quarter     = Column(String(20))
    filed_at    = Column(String(30))
    raw_text    = Column(Text)
    created_at  = Column(DateTime, default=func.now())

class Segment(Base):
    __tablename__ = "segments"

    id             = Column(Integer, primary_key=True, index=True)
    transcript_id  = Column(Integer, index=True)
    ticker         = Column(String(10), index=True)
    quarter        = Column(String(20))
    speaker        = Column(String(200))   # increased from 100
    title          = Column(Text)          # Text instead of VARCHAR
    role           = Column(String(20))
    segment_type   = Column(String(30))
    text           = Column(Text)
    created_at     = Column(DateTime, default=func.now())