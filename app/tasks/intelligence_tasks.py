from app.database import SessionLocal
from app.tasks.celery_app import celery_app
from app.intelligence.entity_extractor import entity_extractor
from app.intelligence.confidence_scorer import confidence_scorer
from app.intelligence.embeddings import get_finbert_encoder
from app.intelligence.report_generator import report_generator


def get_db():
    """Get a raw database session (caller must close it)."""
    return SessionLocal()


@celery_app.task(bind=True)
def process_entities_task(self, transcript_id: int):
    """Background task: entity extraction for a transcript."""
    db = get_db()
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting entity extraction"})

        from app import models
        transcript = db.query(models.Transcript).filter_by(id=transcript_id).first()
        if not transcript:
            raise ValueError(f"Transcript {transcript_id} not found")

        segments = db.query(models.Segment).filter_by(transcript_id=transcript_id).all()
        total = len(segments)
        processed = failed = 0

        for i, segment in enumerate(segments):
            try:
                if entity_extractor.process_segment(db, segment):
                    processed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error on segment {segment.id}: {e}")
                failed += 1

            self.update_state(
                state="PROCESSING",
                meta={"status": f"Processed {i+1}/{total}", "progress": (i+1)/total*100,
                      "processed": processed, "failed": failed}
            )

        db.commit()
        return {"status": "completed", "total_segments": total, "processed": processed, "failed": failed}

    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


@celery_app.task(bind=True)
def process_confidence_task(self, transcript_id: int):
    """Background task: confidence scoring for a transcript."""
    db = get_db()
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting confidence scoring"})

        from app import models
        transcript = db.query(models.Transcript).filter_by(id=transcript_id).first()
        if not transcript:
            raise ValueError(f"Transcript {transcript_id} not found")

        segments = db.query(models.Segment).filter_by(transcript_id=transcript_id).all()
        total = len(segments)
        processed = failed = 0

        for i, segment in enumerate(segments):
            try:
                if confidence_scorer.process_segment(db, segment):
                    processed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error on segment {segment.id}: {e}")
                failed += 1

            self.update_state(
                state="PROCESSING",
                meta={"status": f"Scored {i+1}/{total}", "progress": (i+1)/total*100,
                      "processed": processed, "failed": failed}
            )

        db.commit()
        return {"status": "completed", "total_segments": total, "processed": processed, "failed": failed}

    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


@celery_app.task(bind=True)
def process_embeddings_task(self, transcript_id: int):
    """Background task: FinBERT embeddings for a transcript."""
    db = get_db()
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting embeddings"})

        from app import models
        transcript = db.query(models.Transcript).filter_by(id=transcript_id).first()
        if not transcript:
            raise ValueError(f"Transcript {transcript_id} not found")

        encoder = get_finbert_encoder()
        segments = db.query(models.Segment).filter(
            models.Segment.transcript_id == transcript_id,
            models.Segment.embedding.is_(None)
        ).all()

        total = len(segments)
        if total == 0:
            return {"status": "completed", "total_segments": 0, "processed": 0, "failed": 0,
                    "message": "All segments already have embeddings"}

        processed = failed = 0

        for i, segment in enumerate(segments):
            try:
                if encoder.process_segment(db, segment):
                    processed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error on segment {segment.id}: {e}")
                failed += 1

            self.update_state(
                state="PROCESSING",
                meta={"status": f"Embedded {i+1}/{total}", "progress": (i+1)/total*100,
                      "processed": processed, "failed": failed}
            )

        db.commit()
        return {"status": "completed", "total_segments": total, "processed": processed, "failed": failed}

    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


@celery_app.task
def generate_report_task(transcript_id: int):
    """Background task: generate call report."""
    db = get_db()
    try:
        report = report_generator.save_report(db, transcript_id)
        if report:
            return {"status": "completed", "report_id": report.id,
                    "ticker": report.ticker, "quarter": report.quarter}
        return {"status": "failed", "message": "Failed to generate report"}
    except Exception as e:
        return {"status": "failed", "message": str(e)}
    finally:
        db.close()


@celery_app.task
def process_full_intelligence_task(transcript_id: int):
    """
    Orchestrate full pipeline: entities → confidence → embeddings → report.

    Uses .si() (immutable signatures) so each task receives transcript_id
    directly, not the return value of the previous task.
    """
    from celery import chain

    pipeline = chain(
        process_entities_task.si(transcript_id),
        process_confidence_task.si(transcript_id),
        process_embeddings_task.si(transcript_id),
        generate_report_task.si(transcript_id),
    )

    return pipeline.apply_async()
