from celery import current_task
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.tasks.celery_app import celery_app
from app.intelligence.entity_extractor import entity_extractor
from app.intelligence.confidence_scorer import confidence_scorer
from app.intelligence.embeddings import get_finbert_encoder
from app.intelligence.report_generator import report_generator

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # Don't close here, let the calling function handle it

@celery_app.task(bind=True)
def process_entities_task(self, transcript_id: int):
    """Background task to process entity extraction for a transcript."""
    db = get_db()
    try:
        # Update task status
        self.update_state(state="PROCESSING", meta={"status": "Starting entity extraction"})
        
        # Get transcript
        from app import models
        transcript = db.query(models.Transcript).filter_by(id=transcript_id).first()
        
        if not transcript:
            raise ValueError(f"Transcript {transcript_id} not found")
        
        # Process entities for all segments in this transcript
        segments = db.query(models.Segment).filter_by(transcript_id=transcript_id).all()
        total_segments = len(segments)
        
        processed = 0
        failed = 0
        
        for i, segment in enumerate(segments):
            try:
                success = entity_extractor.process_segment(db, segment)
                if success:
                    processed += 1
                else:
                    failed += 1
                
                # Update progress
                progress = (i + 1) / total_segments * 100
                self.update_state(
                    state="PROCESSING",
                    meta={
                        "status": f"Processed {i+1}/{total_segments} segments",
                        "progress": progress,
                        "processed": processed,
                        "failed": failed
                    }
                )
                
            except Exception as e:
                print(f"Error processing segment {segment.id}: {e}")
                failed += 1
        
        db.commit()
        
        return {
            "status": "completed",
            "total_segments": total_segments,
            "processed": processed,
            "failed": failed
        }
        
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

@celery_app.task(bind=True)
def process_confidence_task(self, transcript_id: int):
    """Background task to process confidence scoring for a transcript."""
    db = get_db()
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting confidence scoring"})
        
        from app import models
        transcript = db.query(models.Transcript).filter_by(id=transcript_id).first()
        
        if not transcript:
            raise ValueError(f"Transcript {transcript_id} not found")
        
        segments = db.query(models.Segment).filter_by(transcript_id=transcript_id).all()
        total_segments = len(segments)
        
        processed = 0
        failed = 0
        
        for i, segment in enumerate(segments):
            try:
                success = confidence_scorer.process_segment(db, segment)
                if success:
                    processed += 1
                else:
                    failed += 1
                
                progress = (i + 1) / total_segments * 100
                self.update_state(
                    state="PROCESSING",
                    meta={
                        "status": f"Scored {i+1}/{total_segments} segments",
                        "progress": progress,
                        "processed": processed,
                        "failed": failed
                    }
                )
                
            except Exception as e:
                print(f"Error scoring segment {segment.id}: {e}")
                failed += 1
        
        db.commit()
        
        return {
            "status": "completed",
            "total_segments": total_segments,
            "processed": processed,
            "failed": failed
        }
        
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

@celery_app.task(bind=True)
def process_embeddings_task(self, transcript_id: int):
    """Background task to process embeddings for a transcript."""
    db = get_db()
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting embeddings generation"})
        
        from app import models
        transcript = db.query(models.Transcript).filter_by(id=transcript_id).first()
        
        if not transcript:
            raise ValueError(f"Transcript {transcript_id} not found")
        
        # Get FinBERT encoder
        encoder = get_finbert_encoder()
        
        segments = db.query(models.Segment).filter_by(
            transcript_id=transcript_id,
            embedding=None  # Only process segments without embeddings
        ).all()
        
        total_segments = len(segments)
        
        if total_segments == 0:
            return {
                "status": "completed",
                "total_segments": 0,
                "processed": 0,
                "failed": 0,
                "message": "All segments already have embeddings"
            }
        
        processed = 0
        failed = 0
        
        for i, segment in enumerate(segments):
            try:
                success = encoder.process_segment(db, segment)
                if success:
                    processed += 1
                else:
                    failed += 1
                
                progress = (i + 1) / total_segments * 100
                self.update_state(
                    state="PROCESSING",
                    meta={
                        "status": f"Generated embeddings for {i+1}/{total_segments} segments",
                        "progress": progress,
                        "processed": processed,
                        "failed": failed
                    }
                )
                
            except Exception as e:
                print(f"Error generating embedding for segment {segment.id}: {e}")
                failed += 1
        
        db.commit()
        
        return {
            "status": "completed",
            "total_segments": total_segments,
            "processed": processed,
            "failed": failed
        }
        
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

@celery_app.task
def generate_report_task(transcript_id: int):
    """Background task to generate call report."""
    db = get_db()
    try:
        report = report_generator.save_report(db, transcript_id)
        
        if report:
            return {
                "status": "completed",
                "report_id": report.id,
                "ticker": report.ticker,
                "quarter": report.quarter
            }
        else:
            return {
                "status": "failed",
                "message": "Failed to generate report"
            }
            
    except Exception as e:
        print(f"Error generating report for transcript {transcript_id}: {e}")
        return {
            "status": "failed",
            "message": str(e)
        }
    finally:
        db.close()

@celery_app.task
def process_full_intelligence_task(transcript_id: int):
    """Chain task to run full intelligence pipeline for a transcript."""
    # This will be the main orchestrator task
    from celery import chain
    
    # Create a chain of tasks: entities -> confidence -> embeddings -> report
    pipeline = chain(
        process_entities_task.s(transcript_id),
        process_confidence_task.s(transcript_id),
        process_embeddings_task.s(transcript_id),
        generate_report_task.s(transcript_id)
    )
    
    return pipeline.apply_async()
