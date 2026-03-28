"""
app/intelligence/drift_calculator.py  — v2

Phase 2 — Drift Detection Engine (recalibrated)

What changed from v1:
─────────────────────
1. RELATIVE THRESHOLDS
   FinBERT embeddings of financial text cluster tightly — cosine distances
   of 0.01-0.015 are normal even between different quarters. Absolute
   thresholds (0.10 / 0.25) don't work here. We now compute thresholds
   dynamically from the score distribution across all quarters for a ticker.

2. TOPIC-FILTERED CENTROIDS (stricter)
   Requires >= MIN_SEGMENTS_FOR_CENTROID segments before forming a centroid.
   A centroid from 1-2 segments is too noisy to be meaningful.

3. COMPOSITE SCORE = centroid_drift + tail_drift
   Instead of only comparing centroids (which averages out signal), we also
   compute the 90th-percentile distance of individual quarter_to segments
   from the quarter_from centroid. A shift in the *tail* — 2–3 paragraphs
   suddenly far from the baseline — is often the real narrative change signal.

   Final drift_score = 0.6 * centroid_distance + 0.4 * tail_distance

4. COMPARE endpoint now returns centroid_dist and tail_dist separately
   so you can see which component is driving a score.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from app import models


# ─── Config ───────────────────────────────────────────────────────────────────

# Minimum segments needed to form a reliable centroid
MIN_SEGMENTS_FOR_CENTROID = 3

# Composite score weights
CENTROID_WEIGHT = 0.6
TAIL_WEIGHT     = 0.4
TAIL_PERCENTILE = 90.0

TOPICS = ["guidance", "risks", "metrics", "overall"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sort_quarters(quarters: List[str]) -> List[str]:
    def _key(q: str):
        try:
            year, qnum = q.split("Q")
            return (int(year), int(qnum))
        except Exception:
            return (0, 0)
    return sorted(quarters, key=_key)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity, clamped to [0, 1]."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    sim = np.dot(a, b) / (norm_a * norm_b)
    return float(np.clip(1.0 - sim, 0.0, 1.0))


def _compute_relative_thresholds(all_scores: List[float]) -> Tuple[float, float]:
    """
    Calibrate thresholds from the actual score distribution for this ticker.

    p75 → stable/drifting boundary  (top 25% are at least "drifting")
    p90 → drifting/sharp_break boundary (top 10% are "sharp_break")

    If fewer than 4 observations, fall back to tight absolute thresholds
    suited to FinBERT's compressed distance space.
    """
    if len(all_scores) < 4:
        return 0.008, 0.012

    arr = np.array(all_scores)
    return float(np.percentile(arr, 75)), float(np.percentile(arr, 90))


def _label(score: float, stable_thresh: float, drifting_thresh: float) -> str:
    if score < stable_thresh:
        return "stable"
    elif score < drifting_thresh:
        return "drifting"
    else:
        return "sharp_break"


# ─── Centroid + distribution builder ─────────────────────────────────────────

def _build_topic_data(
    db: Session, ticker: str
) -> Dict[str, Dict[str, Dict]]:
    """
    Returns:
        {
          topic: {
            quarter: {
              "centroid":   np.ndarray (768,),
              "embeddings": [np.ndarray, ...],
              "count":      int,
            }
          }
        }
    Only quarters with >= MIN_SEGMENTS_FOR_CENTROID segments are included.
    """
    segments = (
        db.query(models.Segment)
        .filter(
            models.Segment.ticker == ticker.upper(),
            models.Segment.embedding.is_not(None),
        )
        .all()
    )

    if not segments:
        return {}

    seg_map: Dict[int, Tuple[models.Segment, np.ndarray]] = {
        s.id: (s, np.array(s.embedding, dtype=np.float32))
        for s in segments
    }

    entity_rows = (
        db.query(models.EntityExtraction.segment_id, models.EntityExtraction.entity_type)
        .filter(models.EntityExtraction.segment_id.in_(list(seg_map.keys())))
        .all()
    )
    seg_topics: Dict[int, set] = {}
    for seg_id, etype in entity_rows:
        seg_topics.setdefault(seg_id, set()).add(etype)

    buckets: Dict[str, Dict[str, List[np.ndarray]]] = {t: {} for t in TOPICS}

    for seg_id, (seg, emb) in seg_map.items():
        q = seg.quarter
        buckets["overall"].setdefault(q, []).append(emb)
        for topic in ["guidance", "risks", "metrics"]:
            if topic in seg_topics.get(seg_id, set()):
                buckets[topic].setdefault(q, []).append(emb)

    result: Dict[str, Dict[str, Dict]] = {}
    for topic, quarter_map in buckets.items():
        result[topic] = {}
        for quarter, embs in quarter_map.items():
            if len(embs) < MIN_SEGMENTS_FOR_CENTROID:
                continue
            stack    = np.stack(embs, axis=0)
            centroid = stack.mean(axis=0)
            result[topic][quarter] = {
                "centroid":   centroid,
                "embeddings": embs,
                "count":      len(embs),
            }

    return result


# ─── Composite scorer ─────────────────────────────────────────────────────────

def _composite_drift(
    centroid_from: np.ndarray,
    centroid_to:   np.ndarray,
    embeddings_to: List[np.ndarray],
) -> Tuple[float, float, float]:
    """
    Returns (composite_score, centroid_dist, tail_dist).

    centroid_dist: cosine distance between centroids
    tail_dist:     TAIL_PERCENTILE-th percentile distance of individual
                   quarter_to segments from the quarter_from centroid
    composite:     weighted blend
    """
    centroid_dist = _cosine_distance(centroid_from, centroid_to)

    if len(embeddings_to) == 0:
        return centroid_dist, centroid_dist, 0.0

    individual_dists = [_cosine_distance(centroid_from, e) for e in embeddings_to]
    tail_dist = float(np.percentile(individual_dists, TAIL_PERCENTILE))

    composite = CENTROID_WEIGHT * centroid_dist + TAIL_WEIGHT * tail_dist
    return composite, centroid_dist, tail_dist


# ─── Main calculator ──────────────────────────────────────────────────────────

class DriftCalculator:

    def calculate_drift(self, db: Session, ticker: str) -> Dict:
        """
        Compute drift scores for all consecutive quarter pairs.
        Thresholds are calibrated relative to this ticker's own distribution.
        """
        ticker = ticker.upper()
        print(f"\n[Drift] Calculating drift for {ticker}...")

        topic_data = _build_topic_data(db, ticker)
        if not topic_data:
            return {"ticker": ticker, "error": "No embedded segments found"}

        # Pass 1: collect all raw scores for threshold calibration
        raw_scores: List[float] = []
        transitions: List[Dict] = []

        for topic, quarter_map in topic_data.items():
            sorted_quarters = _sort_quarters(list(quarter_map.keys()))
            if len(sorted_quarters) < 2:
                continue
            for i in range(len(sorted_quarters) - 1):
                q_from = sorted_quarters[i]
                q_to   = sorted_quarters[i + 1]
                d_from = quarter_map[q_from]
                d_to   = quarter_map[q_to]

                composite, centroid_dist, tail_dist = _composite_drift(
                    d_from["centroid"],
                    d_to["centroid"],
                    d_to["embeddings"],
                )
                raw_scores.append(composite)
                transitions.append({
                    "topic":         topic,
                    "quarter_from":  q_from,
                    "quarter_to":    q_to,
                    "composite":     composite,
                    "centroid_dist": centroid_dist,
                    "tail_dist":     tail_dist,
                    "count_from":    d_from["count"],
                    "count_to":      d_to["count"],
                })

        if not raw_scores:
            return {"ticker": ticker, "error": "Not enough quarters to compute drift (need ≥ 2)"}

        stable_thresh, drifting_thresh = _compute_relative_thresholds(raw_scores)
        print(f"  [Drift] Thresholds — stable<{stable_thresh:.6f}  "
              f"drifting<{drifting_thresh:.6f}  "
              f"sharp_break≥{drifting_thresh:.6f}")

        # Pass 2: label and persist
        db.query(models.DriftScore).filter_by(ticker=ticker).delete()

        rows_created = 0
        results      = []

        for t in transitions:
            lbl  = _label(t["composite"], stable_thresh, drifting_thresh)
            icon = {"stable": "✓", "drifting": "⚠", "sharp_break": "🔴"}.get(lbl, "?")
            print(
                f"    {icon} {t['quarter_from']}→{t['quarter_to']} [{t['topic']:<10}] "
                f"composite={t['composite']:.6f}  "
                f"centroid={t['centroid_dist']:.6f}  "
                f"tail={t['tail_dist']:.6f}  ({lbl})"
            )

            db.add(models.DriftScore(
                ticker             = ticker,
                topic              = t["topic"],
                quarter_from       = t["quarter_from"],
                quarter_to         = t["quarter_to"],
                drift_score        = round(t["composite"], 6),
                label              = lbl,
                segment_count_from = t["count_from"],
                segment_count_to   = t["count_to"],
            ))
            rows_created += 1
            results.append({
                "topic":         t["topic"],
                "quarter_from":  t["quarter_from"],
                "quarter_to":    t["quarter_to"],
                "drift_score":   round(t["composite"], 6),
                "centroid_dist": round(t["centroid_dist"], 6),
                "tail_dist":     round(t["tail_dist"], 6),
                "label":         lbl,
            })

        db.commit()
        print(f"[Drift] Done — {rows_created} scores stored for {ticker}")

        return {
            "ticker":               ticker,
            "drift_scores_created": rows_created,
            "thresholds": {
                "stable":   round(stable_thresh, 6),
                "drifting": round(drifting_thresh, 6),
            },
            "results": results,
        }

    def get_drift_timeline(
        self, db: Session, ticker: str, topic: str = None
    ) -> List[Dict]:
        query = db.query(models.DriftScore).filter_by(ticker=ticker.upper())
        if topic:
            query = query.filter_by(topic=topic)
        rows = query.order_by(
            models.DriftScore.topic,
            models.DriftScore.quarter_from,
        ).all()
        return [
            {
                "topic":              r.topic,
                "quarter_from":       r.quarter_from,
                "quarter_to":         r.quarter_to,
                "drift_score":        r.drift_score,
                "label":              r.label,
                "segment_count_from": r.segment_count_from,
                "segment_count_to":   r.segment_count_to,
            }
            for r in rows
        ]

    def get_alerts(
        self, db: Session, ticker: str, min_label: str = "drifting"
    ) -> List[Dict]:
        label_filter = (
            ["drifting", "sharp_break"]
            if min_label == "drifting"
            else ["sharp_break"]
        )
        rows = (
            db.query(models.DriftScore)
            .filter(
                models.DriftScore.ticker == ticker.upper(),
                models.DriftScore.label.in_(label_filter),
            )
            .order_by(models.DriftScore.drift_score.desc())
            .all()
        )
        return [
            {
                "topic":        r.topic,
                "quarter_from": r.quarter_from,
                "quarter_to":   r.quarter_to,
                "drift_score":  r.drift_score,
                "label":        r.label,
            }
            for r in rows
        ]

    def get_drifted_quotes(
        self,
        db: Session,
        ticker: str,
        topic: str,
        quarter_from: str,
        quarter_to: str,
        top_n: int = 5,
    ) -> List[Dict]:
        """
        Surface the top_n segments in quarter_to furthest from the
        quarter_from centroid — the sentences that changed most.
        """
        ticker     = ticker.upper()
        topic_data = _build_topic_data(db, ticker)
        topic_map  = topic_data.get(topic, {})

        if quarter_from not in topic_map:
            # Fallback: try without MIN_SEGMENTS filter for sparse topics
            segments_from = (
                db.query(models.Segment)
                .filter(
                    models.Segment.ticker  == ticker,
                    models.Segment.quarter == quarter_from,
                    models.Segment.embedding.is_not(None),
                )
                .all()
            )
            if not segments_from:
                return []
            embs_from    = [np.array(s.embedding, dtype=np.float32) for s in segments_from]
            centroid_from = np.stack(embs_from).mean(axis=0)
        else:
            centroid_from = topic_map[quarter_from]["centroid"]

        segments_to = (
            db.query(models.Segment)
            .filter(
                models.Segment.ticker  == ticker,
                models.Segment.quarter == quarter_to,
                models.Segment.embedding.is_not(None),
            )
            .all()
        )

        if topic != "overall":
            entity_ids = {
                row.segment_id
                for row in db.query(models.EntityExtraction.segment_id)
                .filter(
                    models.EntityExtraction.entity_type == topic,
                    models.EntityExtraction.segment_id.in_([s.id for s in segments_to]),
                )
                .all()
            }
            segments_to = [s for s in segments_to if s.id in entity_ids]

        if not segments_to:
            return []

        # Filter out operator boilerplate — these are structurally different
        # from substantive speech so they always score as outliers, drowning
        # out the real signal. Drop segments that are:
        #   • Very short (< 50 chars) — "Thank you." / "Please go ahead."
        #   • From the Operator role with generic connector phrases
        BOILERPLATE_PHRASES = (
            "this concludes",
            "you may now disconnect",
            "your line is open",
            "your next question",
            "your first question",
            "your final question",
            "comes from the line of",
            "comes from vivek",
            "please go ahead",
            "thank you for joining",
            "thank you. our next",
        )

        def _is_boilerplate(seg: models.Segment) -> bool:
            if len(seg.text.strip()) < 150:
                return True
            if "operator" in seg.speaker.lower():
                return True
            text_lower = seg.text.lower()
            return any(phrase in text_lower for phrase in BOILERPLATE_PHRASES)

        substantive = [s for s in segments_to if not _is_boilerplate(s)]

        # If filtering removed everything, fall back to unfiltered (shouldn't happen)
        if not substantive:
            substantive = segments_to

        scored = sorted(
            [
                (s, _cosine_distance(centroid_from, np.array(s.embedding, dtype=np.float32)))
                for s in substantive
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            {
                "segment_id":       s.id,
                "speaker":          s.speaker,
                "role":             s.role,
                "quarter":          s.quarter,
                "drift_distance":   round(dist, 6),
                "confidence_score": s.confidence_score,
                "text":             s.text,
                "text_preview":     s.text[:300],
            }
            for s, dist in scored[:top_n]
        ]

    def compare_quarters(
        self, db: Session, ticker: str, quarter_from: str, quarter_to: str
    ) -> Dict:
        """Side-by-side drift for two specific quarters across all topics."""
        ticker     = ticker.upper()
        topic_data = _build_topic_data(db, ticker)

        comparison = {}
        for topic in TOPICS:
            topic_map = topic_data.get(topic, {})
            if quarter_from not in topic_map or quarter_to not in topic_map:
                comparison[topic] = {"available": False}
                continue

            d_from = topic_map[quarter_from]
            d_to   = topic_map[quarter_to]
            composite, centroid_dist, tail_dist = _composite_drift(
                d_from["centroid"], d_to["centroid"], d_to["embeddings"]
            )
            comparison[topic] = {
                "available":          True,
                "drift_score":        round(composite, 6),
                "centroid_dist":      round(centroid_dist, 6),
                "tail_dist":          round(tail_dist, 6),
                "segment_count_from": d_from["count"],
                "segment_count_to":   d_to["count"],
            }

        return {
            "ticker":       ticker,
            "quarter_from": quarter_from,
            "quarter_to":   quarter_to,
            "comparison":   comparison,
        }


# Global instance
drift_calculator = DriftCalculator()