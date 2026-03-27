import json
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from app import models
from app.intelligence.sarvam_client import sarvam_client

class EntityExtractor:
    def __init__(self):
        self.client = sarvam_client
    
    def extract_entities_from_text(self, text: str) -> Optional[Dict]:
        """Extract entities from a single text segment."""
        response = self.client.extract_entities(text)
        
        if not response:
            return None
            
        try:
            # Parse JSON response
            entities = json.loads(response)
            return entities
        except json.JSONDecodeError:
            print(f"Failed to parse entity extraction response: {response}")
            return None
    
    def process_segment(self, db: Session, segment: models.Segment) -> bool:
        """Process a single segment and save entity extractions."""
        entities = self.extract_entities_from_text(segment.text)
        
        if not entities:
            return False
        
        # Clear existing extractions for this segment
        db.query(models.EntityExtraction).filter_by(segment_id=segment.id).delete()
        
        # Save new extractions
        for entity_type, values in entities.items():
            if entity_type in ["guidance", "risks", "metrics"] and isinstance(values, list):
                for value in values:
                    if value.strip():  # Skip empty values
                        extraction = models.EntityExtraction(
                            segment_id=segment.id,
                            entity_type=entity_type,
                            entity_value=value.strip(),
                            confidence=1.0  # Default confidence for extracted entities
                        )
                        db.add(extraction)
        
        return True
    
    def process_all_segments(self, db: Session, ticker: str = None, quarter: str = None) -> Dict:
        """Process all segments for entity extraction."""
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
                    print(f"✓ Processed segment {segment.id}")
                else:
                    failed += 1
                    print(f"✗ Failed to process segment {segment.id}")
            except Exception as e:
                print(f"Error processing segment {segment.id}: {e}")
                failed += 1
        
        db.commit()
        
        return {
            "total": len(segments),
            "processed": processed,
            "failed": failed
        }
    
    def get_entities_by_type(self, db: Session, entity_type: str, ticker: str = None, quarter: str = None) -> List[Dict]:
        """Retrieve entities of a specific type."""
        query = db.query(models.EntityExtraction).filter_by(entity_type=entity_type)
        
        if ticker or quarter:
            query = query.join(models.Segment, models.EntityExtraction.segment_id == models.Segment.id)
            if ticker:
                query = query.filter(models.Segment.ticker == ticker.upper())
            if quarter:
                query = query.filter(models.Segment.quarter == quarter)
        
        extractions = query.all()
        
        return [
            {
                "segment_id": e.segment_id,
                "entity_value": e.entity_value,
                "confidence": e.confidence
            }
            for e in extractions
        ]

# Global instance
entity_extractor = EntityExtractor()
