from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
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
    confidence_score = Column(Float, nullable=True)
    embedding      = Column(Vector(768))  # FinBERT embedding dimension
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

class DriftScore(Base):
    """
    Stores the cosine drift between two consecutive quarters
    for a given ticker + topic combination.
 
    drift_score is cosine *distance* (1 - cosine_similarity), so:
      0.0  = identical embeddings (no drift)
      1.0  = completely orthogonal (maximum drift)
 
    label thresholds (tunable in drift_calculator.py):
      < 0.10  → stable
      0.10–0.25 → drifting
      > 0.25  → sharp_break
    """
    __tablename__ = "drift_scores"
 
    id                 = Column(Integer, primary_key=True, index=True)
    ticker             = Column(String(10), index=True)
    topic              = Column(String(50), index=True)  # guidance | risks | metrics | overall
    quarter_from       = Column(String(20))              # earlier quarter
    quarter_to         = Column(String(20))              # later quarter
    drift_score        = Column(Float)                   # cosine distance 0.0–1.0
    label              = Column(String(20))              # stable | drifting | sharp_break
    segment_count_from = Column(Integer, default=0)      # segments that contributed
    segment_count_to   = Column(Integer, default=0)
    created_at         = Column(DateTime, default=func.now())