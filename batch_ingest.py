#!/usr/bin/env python3
"""
batch_ingest.py

Ingest a batch of tickers and run the full intelligence + drift pipeline.
Designed to be run daily — 3 tickers/day stays within AlphaVantage free tier.

RESUME SAFE: Re-running the same batch skips already-completed work at every step.
- Ingest:       skips quarters already in DB
- Intelligence: skips segments that already have EntityExtraction rows (entities)
                skips segments where confidence_score is not NULL (confidence)
- Embeddings:   skips segments that already have an embedding
- Drift:        always recalculates (fast, pure math, < 2s)

Usage:
    python batch_ingest.py --batch 1    # GOOGL, AMZN, META
    python batch_ingest.py --batch 2    # TSLA, JPM, V
    python batch_ingest.py --batch 3    # JNJ, WMT, XOM
    python batch_ingest.py --batch 4    # UNH, MA, LLY
    python batch_ingest.py --batch 5    # AVGO, HD, MRK
    python batch_ingest.py --batch 6    # CVX, PEP, COST
    python batch_ingest.py --ticker TSLA
"""

import requests
import time
import argparse
import sys
from datetime import datetime

BASE_URL = "http://localhost:8000"

BATCHES = {
    1: ["GOOGL", "AMZN", "META"],
    2: ["TSLA", "JPM", "V"],
    3: ["JNJ", "WMT", "XOM"],
    4: ["UNH", "MA", "LLY"],
    5: ["AVGO", "HD", "MRK"],
    6: ["CVX", "PEP", "COST"],
}

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def log(msg, colour=RESET):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{colour}[{ts}] {msg}{RESET}")


# ── Step 1: Ingest ─────────────────────────────────────────────────────────────

def ingest_ticker(ticker: str) -> bool:
    log(f"Ingesting {ticker}...", CYAN)
    r = requests.post(f"{BASE_URL}/ingest/{ticker}", timeout=60)
    if r.status_code != 200:
        log(f"  ✗ Ingest failed: {r.status_code} {r.text[:100]}", RED)
        return False
    data = r.json()
    quarters = data.get("ingested", [])
    if quarters:
        log(f"  ✓ {len(quarters)} new quarters ingested", GREEN)
        for q in quarters:
            log(f"      {q['quarter']}: {q['segments']} segments", GREEN)
    else:
        log(f"  ✓ Already up to date — skipping", YELLOW)
    return True


# ── Step 2: Intelligence (direct Python, no HTTP timeout) ─────────────────────

def run_intelligence(ticker: str) -> bool:
    log(f"Intelligence pipeline for {ticker}...", CYAN)

    from app.database import SessionLocal
    from app import models
    from app.intelligence.entity_extractor import entity_extractor
    from app.intelligence.confidence_scorer import confidence_scorer

    db = SessionLocal()
    try:
        all_segments = db.query(models.Segment).filter_by(ticker=ticker.upper()).all()
        if not all_segments:
            log(f"  ✗ No segments found for {ticker}", RED)
            return False

        all_ids = [s.id for s in all_segments]

        # Pre-fetch which segments already have entity extractions
        done_entity_ids = {
            row[0]
            for row in db.query(models.EntityExtraction.segment_id)
            .filter(models.EntityExtraction.segment_id.in_(all_ids))
            .distinct()
            .all()
        }

        # Segments needing entities: no EntityExtraction row at all
        ent_todo = [s for s in all_segments if s.id not in done_entity_ids]

        # Segments needing confidence: confidence_score is NULL
        # (NULL = never scored; 0.5 default is set at DB level so
        #  we use None check which catches truly unscored rows)
        conf_todo = [s for s in all_segments if s.confidence_score is None]

        log(
            f"  {len(all_segments)} total — "
            f"{len(ent_todo)} need entities, "
            f"{len(conf_todo)} need confidence",
            CYAN
        )

        if not ent_todo and not conf_todo:
            log(f"  ✓ All segments already processed — skipping", YELLOW)
            return True

        # Work through the union of segments that need at least one step
        todo_ids  = {s.id for s in ent_todo} | {s.id for s in conf_todo}
        todo_segs = [s for s in all_segments if s.id in todo_ids]
        total     = len(todo_segs)

        ent_processed = ent_failed = 0
        conf_processed = conf_failed = 0

        for i, seg in enumerate(todo_segs):

            # Entities
            if seg.id not in done_entity_ids:
                try:
                    if entity_extractor.process_segment(db, seg):
                        ent_processed += 1
                    else:
                        ent_failed += 1
                except Exception as e:
                    log(f"  entity error seg {seg.id}: {e}", RED)
                    ent_failed += 1

            # Confidence
            if seg.confidence_score is None:
                try:
                    if confidence_scorer.process_segment(db, seg):
                        conf_processed += 1
                    else:
                        conf_failed += 1
                except Exception as e:
                    log(f"  confidence error seg {seg.id}: {e}", RED)
                    conf_failed += 1

            if (i + 1) % 10 == 0 or (i + 1) == total:
                db.commit()
                log(
                    f"  [{i+1}/{total}] "
                    f"entities={ent_processed} conf={conf_processed} "
                    f"failed={ent_failed + conf_failed}",
                    CYAN
                )

        db.commit()
        log(f"  ✓ Entities — {ent_processed} done, {ent_failed} failed", GREEN)
        log(f"  ✓ Confidence — {conf_processed} done, {conf_failed} failed", GREEN)
        return True

    except Exception as e:
        db.rollback()
        log(f"  ✗ Intelligence failed: {e}", RED)
        return False
    finally:
        db.close()


# ── Step 3: Embeddings (direct Python, no HTTP timeout) ───────────────────────

def run_embeddings(ticker: str) -> bool:
    log(f"FinBERT embeddings for {ticker}...", CYAN)

    from app.database import SessionLocal
    from app.intelligence.embeddings import get_finbert_encoder

    db = SessionLocal()
    try:
        encoder = get_finbert_encoder()
        # process_all_segments already skips segments with existing embeddings
        result = encoder.process_all_segments(db, ticker=ticker.upper())
        if result["total"] == 0:
            log(f"  ✓ All embeddings already exist — skipping", YELLOW)
        else:
            log(
                f"  ✓ {result['processed']} embedded, {result['failed']} failed",
                GREEN
            )
        return True
    except Exception as e:
        db.rollback()
        log(f"  ✗ Embeddings failed: {e}", RED)
        return False
    finally:
        db.close()


# ── Step 4: Drift ──────────────────────────────────────────────────────────────

def run_drift(ticker: str) -> bool:
    log(f"Drift calculation for {ticker}...", CYAN)
    r = requests.post(f"{BASE_URL}/drift/calculate/{ticker}", timeout=60)
    if r.status_code != 200:
        log(f"  ✗ Drift failed: {r.status_code}", RED)
        return False
    d = r.json()
    n      = d.get("drift_scores_created", 0)
    alerts = [x for x in d.get("results", []) if x["label"] != "stable"]
    log(f"  ✓ {n} scores computed, {len(alerts)} alert(s)", GREEN)
    for a in alerts:
        colour = RED if a["label"] == "sharp_break" else YELLOW
        log(
            f"      {colour}[{a['label']}] {a['topic']} "
            f"{a['quarter_from']}→{a['quarter_to']} "
            f"score={a['drift_score']:.4f}{RESET}"
        )
    return True


# ── Orchestrator ───────────────────────────────────────────────────────────────

def process_ticker(ticker: str) -> bool:
    print()
    log(f"{'='*50}", BOLD)
    log(f"  {ticker}", BOLD)
    log(f"{'='*50}", BOLD)

    steps = [
        ("ingest",       lambda: ingest_ticker(ticker)),
        ("intelligence", lambda: run_intelligence(ticker)),
        ("embeddings",   lambda: run_embeddings(ticker)),
        ("drift",        lambda: run_drift(ticker)),
    ]

    for step_name, fn in steps:
        if not fn():
            log(f"Stopped at '{step_name}' for {ticker}", RED)
            return False
        time.sleep(2)

    log(f"✓ {ticker} complete", GREEN)
    return True


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch", type=int, choices=BATCHES.keys())
    group.add_argument("--ticker", type=str)
    args = parser.parse_args()

    try:
        assert requests.get(f"{BASE_URL}/", timeout=5).status_code == 200
    except Exception:
        print(f"{RED}Server not running: uvicorn app.main:app --reload{RESET}")
        sys.exit(1)

    tickers = [args.ticker.upper()] if args.ticker else BATCHES[args.batch]

    print(f"\n{BOLD}Batch Ingestion — {datetime.now().strftime('%Y-%m-%d %H:%M')}{RESET}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Resume-safe: already-processed segments are skipped at every step\n")

    results = {}
    for i, ticker in enumerate(tickers):
        results[ticker] = process_ticker(ticker)
        if i < len(tickers) - 1:
            log(f"Waiting 15s before next ticker...", YELLOW)
            time.sleep(15)

    print(f"\n{BOLD}Summary{RESET}")
    for ticker, ok in results.items():
        icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {icon} {ticker}")

    failed = [t for t, ok in results.items() if not ok]
    if failed:
        print(f"\n{YELLOW}Retry with:{RESET}")
        for t in failed:
            print(f"  python batch_ingest.py --ticker {t}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}All done.{RESET}")
        print(f"Alerts: GET {BASE_URL}/drift/summary")


if __name__ == "__main__":
    main()