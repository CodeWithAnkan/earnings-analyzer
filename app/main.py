from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db, engine
from app import models
from app.ingestion.alphavantage import fetch_all_transcripts

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
        raise HTTPException(
            status_code=404,
            detail=f"No transcripts found for {ticker}"
        )

    ingested = []

    for data in all_transcripts:
        # Skip if already ingested
        existing = db.query(models.Transcript).filter_by(
            ticker=ticker.upper(),
            quarter=data["quarter"]
        ).first()
        if existing:
            print(f"  Skipping {data['quarter']} — already ingested")
            continue

        # Save transcript record
        transcript = models.Transcript(
            ticker=ticker.upper(),
            quarter=data["quarter"],
            filed_at=data["quarter"],
            raw_text=f"{ticker} {data['quarter']} earnings call"
        )
        db.add(transcript)
        db.flush()

        # Save each speaker segment
        for seg in data["segments"]:
            speaker = seg.get("speaker", "Unknown")
            title = seg.get("title", "")
            content = seg.get("content", "")
            sentiment = seg.get("sentiment", "0.0")

            if not content or len(content) < 20:
                continue

            # Determine role
            role = "management"
            title_lower = title.lower()
            speaker_lower = speaker.lower()
            if any(word in title_lower or word in speaker_lower for word in [
                "analyst", "research", "capital", "securities",
                "partners", "bank", "operator"
            ]):
                role = "analyst"

            # Determine segment type based on position
            # Operator introducing Q&A is a reliable marker
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

        ingested.append({
            "quarter": data["quarter"],
            "segments": len(data["segments"])
        })

    db.commit()
    return {"ticker": ticker.upper(), "ingested": ingested}

@app.get("/segments/{ticker}")
def get_segments(ticker: str, role: str = None, db: Session = Depends(get_db)):
    """View all parsed segments for a ticker. Filter by role=management or role=analyst."""
    query = db.query(models.Segment).filter_by(ticker=ticker.upper())
    if role:
        query = query.filter_by(role=role)
    segs = query.order_by(models.Segment.quarter.desc()).all()
    return [
        {
            "quarter": s.quarter,
            "speaker": s.speaker,
            "title": s.title,
            "role": s.role,
            "type": s.segment_type,
            "preview": s.text[:200]
        }
        for s in segs
    ]

@app.get("/segments/{ticker}/{quarter}")
def get_quarter_segments(ticker: str, quarter: str, db: Session = Depends(get_db)):
    """View segments for a specific quarter. E.g. /segments/NVDA/2025Q3"""
    segs = db.query(models.Segment).filter_by(
        ticker=ticker.upper(),
        quarter=quarter
    ).all()
    if not segs:
        raise HTTPException(status_code=404, detail="No segments found")
    return [
        {
            "speaker": s.speaker,
            "title": s.title,
            "role": s.role,
            "type": s.segment_type,
            "text": s.text
        }
        for s in segs
    ]