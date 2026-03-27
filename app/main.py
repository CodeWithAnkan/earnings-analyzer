from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db, engine
from app import models
from app.ingestion.alphavantage import fetch_all_transcripts
from app.intelligence.entity_extractor import entity_extractor
from app.intelligence.confidence_scorer import confidence_scorer
from app.intelligence.embeddings import get_finbert_encoder
from app.intelligence.report_generator import report_generator

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Earnings Analyzer")


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/ingest/{ticker}")
def ingest_ticker(ticker: str, db: Session = Depends(get_db)):
    """Ingest 8 quarters of earnings call transcripts for a ticker."""
    print(f"\nFetching transcripts for {ticker}...")
    all_transcripts = fetch_all_transcripts(ticker.upper())

    if not all_transcripts:
        raise HTTPException(status_code=404, detail=f"No transcripts found for {ticker}")

    ingested = []

    for data in all_transcripts:
        existing = db.query(models.Transcript).filter_by(
            ticker=ticker.upper(), quarter=data["quarter"]
        ).first()
        if existing:
            print(f"  Skipping {data['quarter']} — already ingested")
            continue

        transcript = models.Transcript(
            ticker=ticker.upper(),
            quarter=data["quarter"],
            filed_at=data["quarter"],
            raw_text=f"{ticker} {data['quarter']} earnings call"
        )
        db.add(transcript)
        db.flush()

        for seg in data["segments"]:
            speaker = seg.get("speaker", "Unknown")
            title = seg.get("title", "")
            content = seg.get("content", "")

            if not content or len(content) < 20:
                continue

            role = "management"
            if any(word in title.lower() or word in speaker.lower() for word in [
                "analyst", "research", "capital", "securities", "partners", "bank", "operator"
            ]):
                role = "analyst"

            segment_type = "prepared_remarks"
            if "question" in content.lower()[:100] or "q&a" in content.lower()[:100]:
                segment_type = "qa"

            db.add(models.Segment(
                transcript_id=transcript.id,
                ticker=ticker.upper(),
                quarter=data["quarter"],
                speaker=speaker[:200],
                title=title[:500],
                role=role,
                segment_type=segment_type,
                text=content
            ))

        ingested.append({"quarter": data["quarter"], "segments": len(data["segments"])})

    db.commit()
    return {"ticker": ticker.upper(), "ingested": ingested}


@app.get("/segments/{ticker}")
def get_segments(ticker: str, role: str = None, db: Session = Depends(get_db)):
    query = db.query(models.Segment).filter_by(ticker=ticker.upper())
    if role:
        query = query.filter_by(role=role)
    segs = query.order_by(models.Segment.quarter.desc()).all()
    return [
        {"quarter": s.quarter, "speaker": s.speaker, "title": s.title,
         "role": s.role, "type": s.segment_type, "preview": s.text[:200]}
        for s in segs
    ]


@app.get("/segments/{ticker}/{quarter}")
def get_quarter_segments(ticker: str, quarter: str, db: Session = Depends(get_db)):
    segs = db.query(models.Segment).filter_by(
        ticker=ticker.upper(), quarter=quarter
    ).all()
    if not segs:
        raise HTTPException(status_code=404, detail="No segments found")
    return [
        {"speaker": s.speaker, "title": s.title, "role": s.role,
         "type": s.segment_type, "text": s.text}
        for s in segs
    ]


# ─── Intelligence endpoints ───────────────────────────────────────────────────

@app.post("/intelligence/entities/{ticker}")
def process_entities(ticker: str, quarter: str = None, db: Session = Depends(get_db)):
    return entity_extractor.process_all_segments(db, ticker=ticker.upper(), quarter=quarter)


@app.post("/intelligence/confidence/{ticker}")
def process_confidence(ticker: str, quarter: str = None, db: Session = Depends(get_db)):
    return confidence_scorer.process_all_segments(db, ticker=ticker.upper(), quarter=quarter)


@app.post("/intelligence/embeddings/{ticker}")
def process_embeddings(ticker: str, quarter: str = None, db: Session = Depends(get_db)):
    encoder = get_finbert_encoder()
    return encoder.process_all_segments(db, ticker=ticker.upper(), quarter=quarter)


@app.post("/intelligence/full/{ticker}")
def process_full_intelligence(ticker: str, quarter: str = None, db: Session = Depends(get_db)):
    """Queue full intelligence pipeline via Celery. Requires Redis + worker running."""
    # Lazy import — Celery/Redis only needed when this endpoint is actually called
    from app.tasks.intelligence_tasks import process_full_intelligence_task

    query = db.query(models.Transcript).filter_by(ticker=ticker.upper())
    if quarter:
        query = query.filter_by(quarter=quarter)
    transcripts = query.all()

    if not transcripts:
        raise HTTPException(status_code=404, detail="No transcripts found")

    task_ids = []
    for transcript in transcripts:
        task = process_full_intelligence_task.delay(transcript.id)
        task_ids.append({
            "transcript_id": transcript.id,
            "quarter": transcript.quarter,
            "task_id": task.id
        })

    return {"ticker": ticker.upper(), "tasks_queued": len(task_ids), "task_ids": task_ids}


@app.get("/intelligence/entities/{ticker}")
def get_entities(ticker: str, entity_type: str, quarter: str = None, db: Session = Depends(get_db)):
    if entity_type not in ["guidance", "risks", "metrics"]:
        raise HTTPException(status_code=400, detail="entity_type must be guidance, risks, or metrics")
    entities = entity_extractor.get_entities_by_type(db, entity_type, ticker=ticker.upper(), quarter=quarter)
    return {"ticker": ticker.upper(), "entity_type": entity_type, "entities": entities}


@app.get("/intelligence/confidence/{ticker}")
def get_confidence_stats(ticker: str, quarter: str = None, db: Session = Depends(get_db)):
    stats = confidence_scorer.get_confidence_stats(db, ticker=ticker.upper(), quarter=quarter)
    return {"ticker": ticker.upper(), "quarter": quarter, "confidence_stats": stats}


@app.get("/intelligence/low-confidence/{ticker}")
def get_low_confidence_segments(ticker: str, threshold: float = 0.3, quarter: str = None, db: Session = Depends(get_db)):
    segments = confidence_scorer.get_low_confidence_segments(db, threshold, ticker=ticker.upper(), quarter=quarter)
    return {"ticker": ticker.upper(), "threshold": threshold, "segments": segments}


@app.get("/intelligence/similar")
def find_similar_segments(query: str, ticker: str = None, limit: int = 10, db: Session = Depends(get_db)):
    encoder = get_finbert_encoder()
    similar = encoder.find_similar_segments(db, query, ticker=ticker.upper() if ticker else None, limit=limit)
    return {"query": query, "ticker": ticker, "similar_segments": similar}


@app.get("/intelligence/embeddings/stats/{ticker}")
def get_embedding_stats(ticker: str, quarter: str = None, db: Session = Depends(get_db)):
    encoder = get_finbert_encoder()
    stats = encoder.get_embedding_stats(db, ticker=ticker.upper(), quarter=quarter)
    return {"ticker": ticker.upper(), "embedding_stats": stats}


@app.post("/intelligence/reports/{ticker}/{quarter}")
def generate_report(ticker: str, quarter: str, db: Session = Depends(get_db)):
    transcript = db.query(models.Transcript).filter_by(
        ticker=ticker.upper(), quarter=quarter
    ).first()
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")
    report = report_generator.save_report(db, transcript.id)
    if not report:
        raise HTTPException(status_code=500, detail="Failed to generate report")
    return {"ticker": ticker.upper(), "quarter": quarter, "report_id": report.id, "generated_at": str(report.created_at)}


@app.get("/intelligence/reports/{ticker}/{quarter}")
def get_report(ticker: str, quarter: str, db: Session = Depends(get_db)):
    report = report_generator.get_report(db, ticker.upper(), quarter)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@app.get("/intelligence/tasks/{task_id}")
def get_task_status(task_id: str):
    """Check status of a background Celery task."""
    from app.tasks.celery_app import celery_app

    result = celery_app.AsyncResult(task_id)

    if result.state == "PENDING":
        return {"state": result.state, "status": "Task is waiting to be processed"}
    elif result.state == "PROCESSING":
        return {
            "state": result.state,
            "status": result.info.get("status", ""),
            "progress": result.info.get("progress", 0),
            "processed": result.info.get("processed", 0),
            "failed": result.info.get("failed", 0),
        }
    elif result.state == "SUCCESS":
        return {"state": result.state, "result": result.result}
    else:
        return {"state": result.state, "error": str(result.info)}
