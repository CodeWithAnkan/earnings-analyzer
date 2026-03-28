"""
app/tasks/drift_tasks.py

Celery tasks for Phase 2 drift detection.
Add this file alongside intelligence_tasks.py.
Also add the import to celery_app.py include list.
"""

from app.database import SessionLocal
from app.tasks.celery_app import celery_app
from app.intelligence.drift_calculator import drift_calculator


def get_db():
    return SessionLocal()


@celery_app.task(bind=True)
def calculate_drift_task(self, ticker: str):
    """
    Background task: compute drift scores for all consecutive quarter pairs
    for a given ticker.
    """
    db = get_db()
    try:
        self.update_state(
            state="PROCESSING",
            meta={"status": f"Calculating drift for {ticker}..."}
        )
        result = drift_calculator.calculate_drift(db, ticker)
        return result
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


@celery_app.task(bind=True)
def calculate_drift_multi_task(self, tickers: list):
    """
    Background task: compute drift scores for multiple tickers sequentially.
    Returns a combined summary.
    """
    db = get_db()
    results = []
    try:
        for i, ticker in enumerate(tickers):
            self.update_state(
                state="PROCESSING",
                meta={
                    "status": f"Processing {ticker} ({i+1}/{len(tickers)})",
                    "progress": (i / len(tickers)) * 100,
                    "current_ticker": ticker,
                }
            )
            result = drift_calculator.calculate_drift(db, ticker)
            results.append(result)
        return {"status": "completed", "tickers": tickers, "results": results}
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()