from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import VECTOR
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
    confidence_score = Column(Float, default=0.5)
    embedding      = Column(VECTOR(768))  # FinBERT embedding dimension
    created_at     = Column(DateTime, default=func.now())

class EntityExtraction(Base):
    __tablename__ = "entity_extractions"

    id          = Column(Integer, primary_key=True, index=True)
    segment_id  = Column(Integer, ForeignKey("segments.id"), index=True)
    entity_type = Column(String(50))  # guidance, risks, metrics
    entity_value = Column(Text)
    confidence  = Column(Float, default=0.0)
    created_at  = Column(DateTime, default=func.now())

class CallReport(Base):
    __tablename__ = "call_reports"

    id          = Column(Integer, primary_key=True, index=True)
    transcript_id = Column(Integer, ForeignKey("transcripts.id"), index=True)
    ticker      = Column(String(10), index=True)
    quarter     = Column(String(20))
    report_json = Column(Text)  # Structured JSON report
    created_at  = Column(DateTime, default=func.now())