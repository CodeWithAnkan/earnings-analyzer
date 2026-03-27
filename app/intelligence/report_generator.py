import json
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from app import models
from app.intelligence.entity_extractor import entity_extractor
from app.intelligence.confidence_scorer import confidence_scorer

class ReportGenerator:
    def __init__(self):
        self.entity_extractor = entity_extractor
        self.confidence_scorer = confidence_scorer
    
    def generate_call_report(self, db: Session, transcript_id: int) -> Optional[Dict]:
        """Generate comprehensive report for a single earnings call."""
        transcript = db.query(models.Transcript).filter_by(id=transcript_id).first()
        
        if not transcript:
            return None
        
        # Get all segments for this transcript
        segments = db.query(models.Segment).filter_by(transcript_id=transcript_id).all()
        
        if not segments:
            return None
        
        # Collect entities from all segments
        all_entities = {"guidance": [], "risks": [], "metrics": []}
        
        for segment in segments:
            entities = db.query(models.EntityExtraction).filter_by(segment_id=segment.id).all()
            for entity in entities:
                if entity.entity_type in all_entities:
                    all_entities[entity.entity_type].append({
                        "value": entity.entity_value,
                        "confidence": entity.confidence,
                        "speaker": segment.speaker,
                        "role": segment.role
                    })
        
        # Calculate confidence statistics
        confidences = [s.confidence_score for s in segments if s.confidence_score is not None]
        confidence_stats = {
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "min_confidence": min(confidences) if confidences else 0.0,
            "max_confidence": max(confidences) if confidences else 0.0,
            "total_segments": len(segments),
            "scored_segments": len(confidences)
        }
        
        # Segment breakdown by role
        role_breakdown = {}
        for segment in segments:
            role = segment.role or "unknown"
            if role not in role_breakdown:
                role_breakdown[role] = {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "segments": []
                }
            
            role_breakdown[role]["count"] += 1
            role_breakdown[role]["segments"].append({
                "speaker": segment.speaker,
                "confidence": segment.confidence_score,
                "text_preview": segment.text[:200]
            })
        
        # Calculate average confidence per role
        for role, data in role_breakdown.items():
            role_confidences = [s["confidence"] for s in data["segments"] if s["confidence"] is not None]
            data["avg_confidence"] = sum(role_confidences) / len(role_confidences) if role_confidences else 0.0
        
        # Compile report
        report = {
            "transcript_info": {
                "ticker": transcript.ticker,
                "quarter": transcript.quarter,
                "filed_at": transcript.filed_at
            },
            "summary": {
                "total_segments": len(segments),
                "confidence_stats": confidence_stats,
                "role_breakdown": role_breakdown
            },
            "entities": all_entities,
            "key_insights": self._generate_key_insights(all_entities, confidence_stats, role_breakdown),
            "generated_at": str(models.func.now())
        }
        
        return report
    
    def _generate_key_insights(self, entities: Dict, confidence_stats: Dict, role_breakdown: Dict) -> List[str]:
        """Generate key insights from the data."""
        insights = []
        
        # Entity-based insights
        if entities["guidance"]:
            insights.append(f"Found {len(entities['guidance'])} guidance statements")
        
        if entities["risks"]:
            insights.append(f"Identified {len(entities['risks'])} risk factors")
        
        if entities["metrics"]:
            insights.append(f"Extracted {len(entities['metrics'])} specific metrics")
        
        # Confidence-based insights
        avg_conf = confidence_stats["avg_confidence"]
        if avg_conf < 0.3:
            insights.append("Low overall confidence suggests uncertainty in management communication")
        elif avg_conf > 0.7:
            insights.append("High overall confidence indicates strong, decisive communication")
        
        # Role-based insights
        if "management" in role_breakdown and "analyst" in role_breakdown:
            mgmt_conf = role_breakdown["management"]["avg_confidence"]
            analyst_conf = role_breakdown["analyst"]["avg_confidence"]
            
            if mgmt_conf > analyst_conf + 0.2:
                insights.append("Management shows significantly higher confidence than analysts")
            elif analyst_conf > mgmt_conf + 0.2:
                insights.append("Analysts show surprisingly high confidence compared to management")
        
        return insights
    
    def save_report(self, db: Session, transcript_id: int) -> Optional[models.CallReport]:
        """Generate and save report to database."""
        report = self.generate_call_report(db, transcript_id)
        
        if not report:
            return None
        
        # Get transcript info for the report record
        transcript = db.query(models.Transcript).filter_by(id=transcript_id).first()
        
        # Delete existing report for this transcript
        existing = db.query(models.CallReport).filter_by(transcript_id=transcript_id).first()
        if existing:
            db.delete(existing)
        
        # Save new report
        call_report = models.CallReport(
            transcript_id=transcript_id,
            ticker=transcript.ticker,
            quarter=transcript.quarter,
            report_json=json.dumps(report, indent=2)
        )
        
        db.add(call_report)
        db.commit()
        
        return call_report
    
    def get_report(self, db: Session, ticker: str, quarter: str) -> Optional[Dict]:
        """Retrieve saved report for a ticker and quarter."""
        report = db.query(models.CallReport).filter_by(
            ticker=ticker.upper(),
            quarter=quarter
        ).first()
        
        if not report:
            return None
        
        return json.loads(report.report_json)

# Global instance
report_generator = ReportGenerator()
