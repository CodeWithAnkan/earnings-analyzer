import numpy as np
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from app import models


class FinBERTEncoder:
    def __init__(self):
        print("Loading FinBERT model...")
        self.model = SentenceTransformer("ProsusAI/finbert")
        self.embedding_dim = 768
        print("FinBERT model loaded successfully!")

    def encode_text(self, text: str) -> Optional[np.ndarray]:
        try:
            return self.model.encode(text, convert_to_numpy=True)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def encode_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=8)
            return list(embeddings)
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return [None] * len(texts)

    def process_segment(self, db: Session, segment: models.Segment) -> bool:
        if segment.embedding is not None:
            return True

        embedding = self.encode_text(segment.text)
        if embedding is None:
            return False

        segment.embedding = embedding.tolist()
        db.add(segment)
        return True

    def process_all_segments(
        self, db: Session, ticker: str = None, quarter: str = None, batch_size: int = 50
    ) -> Dict:
        query = db.query(models.Segment).filter(models.Segment.embedding.is_(None))

        if ticker:
            query = query.filter_by(ticker=ticker.upper())
        if quarter:
            query = query.filter_by(quarter=quarter)

        segments = query.all()

        if not segments:
            return {"total": 0, "processed": 0, "failed": 0}

        processed = 0
        failed = 0

        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            texts = [s.text for s in batch]
            embeddings = self.encode_batch(texts)

            for segment, embedding in zip(batch, embeddings):
                try:
                    if embedding is not None:
                        segment.embedding = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                        processed += 1
                        print(f"✓ Embedded segment {segment.id}")
                    else:
                        failed += 1
                        print(f"✗ Failed segment {segment.id}")
                except Exception as e:
                    print(f"Error on segment {segment.id}: {e}")
                    failed += 1

            db.commit()
            print(f"Batch {i // batch_size + 1} done")

        return {"total": len(segments), "processed": processed, "failed": failed}

    def find_similar_segments(
        self, db: Session, query_text: str, ticker: str = None, limit: int = 10
    ) -> List[Dict]:
        query_embedding = self.encode_text(query_text)
        if query_embedding is None:
            return []

        query = db.query(models.Segment).filter(models.Segment.embedding.is_not(None))
        if ticker:
            query = query.filter_by(ticker=ticker.upper())

        segments = query.all()

        similarities = []
        for segment in segments:
            if segment.embedding is not None:
                seg_emb = np.array(segment.embedding)
                sim = np.dot(query_embedding, seg_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(seg_emb)
                )
                similarities.append((segment, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "id": s.id,
                "ticker": s.ticker,
                "quarter": s.quarter,
                "speaker": s.speaker,
                "role": s.role,
                "similarity": sim,
                "text_preview": s.text[:200],
                "confidence_score": s.confidence_score,
            }
            for s, sim in similarities[:limit]
        ]

    def get_embedding_stats(self, db: Session, ticker: str = None, quarter: str = None) -> Dict:
        embedded_q = db.query(models.Segment).filter(models.Segment.embedding.is_not(None))
        total_q = db.query(models.Segment)

        if ticker:
            embedded_q = embedded_q.filter_by(ticker=ticker.upper())
            total_q = total_q.filter_by(ticker=ticker.upper())
        if quarter:
            embedded_q = embedded_q.filter_by(quarter=quarter)
            total_q = total_q.filter_by(quarter=quarter)

        embedded_count = embedded_q.count()
        total_count = total_q.count()

        return {
            "total_segments": total_count,
            "embedded_segments": embedded_count,
            "coverage_percentage": (embedded_count / total_count * 100) if total_count > 0 else 0,
            "embedding_dimension": self.embedding_dim,
        }


# Global instance (lazy loading)
finbert_encoder = None


def get_finbert_encoder() -> FinBERTEncoder:
    global finbert_encoder
    if finbert_encoder is None:
        finbert_encoder = FinBERTEncoder()
    return finbert_encoder
