from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from app import models
from app.intelligence.sarvam_client import sarvam_client


class ConfidenceScorer:
    def __init__(self):
        self.client = sarvam_client

    def score_segment_confidence(self, text: str) -> Optional[float]:
        """Score confidence for a single text segment."""
        return self.client.score_confidence(text)

    def process_segment(self, db: Session, segment: models.Segment) -> bool:
        """Process a single segment and update confidence score."""
        confidence = self.score_segment_confidence(segment.text)

        if confidence is None:
            return False

        segment.confidence_score = confidence
        db.add(segment)
        return True

    def process_all_segments(self, db: Session, ticker: str = None, quarter: str = None) -> Dict:
        """Process all segments for confidence scoring."""
        query = db.query(models.Segment)

        if ticker:
            query = query.filter_by(ticker=ticker.upper())
        if quarter:
            query = query.filter_by(quarter=quarter)

        segments = query.all()
        processed = 0
        failed = 0

        for segment in segments:
            try:
                if self.process_segment(db, segment):
                    processed += 1
                    print(f"✓ Scored segment {segment.id}: {segment.confidence_score}")
                else:
                    failed += 1
                    print(f"✗ Failed to score segment {segment.id}")
            except Exception as e:
                print(f"Error scoring segment {segment.id}: {e}")
                failed += 1

        db.commit()
        return {"total": len(segments), "processed": processed, "failed": failed}

    def get_confidence_stats(self, db: Session, ticker: str = None, quarter: str = None) -> Dict:
        """Get confidence statistics for segments."""
        query = db.query(models.Segment)

        if ticker:
            query = query.filter_by(ticker=ticker.upper())
        if quarter:
            query = query.filter_by(quarter=quarter)

        segments = query.all()

        if not segments:
            return {"count": 0, "avg_confidence": 0.0, "min_confidence": 0.0, "max_confidence": 0.0}

        confidences = [s.confidence_score for s in segments if s.confidence_score is not None]

        if not confidences:
            return {"count": len(segments), "avg_confidence": 0.0, "min_confidence": 0.0, "max_confidence": 0.0}

        return {
            "count": len(confidences),
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
        }

    def get_low_confidence_segments(
        self, db: Session, threshold: float = 0.3, ticker: str = None, quarter: str = None
    ) -> List[Dict]:
        """Get segments with confidence below threshold."""
        query = db.query(models.Segment).filter(models.Segment.confidence_score < threshold)

        if ticker:
            query = query.filter_by(ticker=ticker.upper())
        if quarter:
            query = query.filter_by(quarter=quarter)

        segments = query.order_by(models.Segment.confidence_score.asc()).all()

        return [
            {
                "id": s.id,
                "speaker": s.speaker,
                "role": s.role,
                "confidence_score": s.confidence_score,
                "text_preview": s.text[:200],
            }
            for s in segments
        ]


# Global instance
confidence_scorer = ConfidenceScorer()
