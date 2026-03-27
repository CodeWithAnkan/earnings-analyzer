import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from app import models

class FinBERTEncoder:
    def __init__(self):
        """Initialize FinBERT model for embeddings."""
        print("Loading FinBERT model...")
        # Using a financial BERT model from sentence-transformers
        self.model = SentenceTransformer('ProsusAI/finbert')
        self.embedding_dim = 768  # FinBERT embedding dimension
        print("FinBERT model loaded successfully!")
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for a single text."""
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def encode_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Generate embeddings for multiple texts efficiently."""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=8)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return [None] * len(texts)
    
    def process_segment(self, db: Session, segment: models.Segment) -> bool:
        """Process a single segment and generate embedding."""
        if segment.embedding is not None:
            print(f"Segment {segment.id} already has embedding")
            return True
        
        embedding = self.encode_text(segment.text)
        
        if embedding is None:
            return False
        
        # Convert numpy array to list for PostgreSQL
        segment.embedding = embedding.tolist()
        db.add(segment)
        
        return True
    
    def process_all_segments(self, db: Session, ticker: str = None, quarter: str = None, batch_size: int = 50) -> Dict:
        """Process all segments for embeddings in batches."""
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
        
        # Process in batches for efficiency
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            texts = [segment.text for segment in batch]
            embeddings = self.encode_batch(texts)
            
            for segment, embedding in zip(batch, embeddings):
                try:
                    if embedding is not None:
                        segment.embedding = embedding.tolist()
                        processed += 1
                        print(f"✓ Processed segment {segment.id}")
                    else:
                        failed += 1
                        print(f"✗ Failed to process segment {segment.id}")
                except Exception as e:
                    print(f"Error processing segment {segment.id}: {e}")
                    failed += 1
            
            # Commit batch
            db.commit()
            print(f"Batch {i//batch_size + 1} completed")
        
        return {
            "total": len(segments),
            "processed": processed,
            "failed": failed
        }
    
    def find_similar_segments(self, db: Session, query_text: str, ticker: str = None, limit: int = 10) -> List[Dict]:
        """Find segments similar to query text using vector similarity."""
        # Generate embedding for query text
        query_embedding = self.encode_text(query_text)
        
        if query_embedding is None:
            return []
        
        # Build query for similar segments
        query = db.query(models.Segment).filter(models.Segment.embedding.is_not(None))
        
        if ticker:
            query = query.filter_by(ticker=ticker.upper())
        
        segments = query.all()
        
        # Calculate similarities
        similarities = []
        for segment in segments:
            if segment.embedding:
                segment_embedding = np.array(segment.embedding)
                similarity = np.dot(query_embedding, segment_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(segment_embedding)
                )
                similarities.append((segment, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for segment, similarity in similarities[:limit]:
            results.append({
                "id": segment.id,
                "ticker": segment.ticker,
                "quarter": segment.quarter,
                "speaker": segment.speaker,
                "role": segment.role,
                "similarity": float(similarity),
                "text_preview": segment.text[:200],
                "confidence_score": segment.confidence_score
            })
        
        return results
    
    def get_embedding_stats(self, db: Session, ticker: str = None, quarter: str = None) -> Dict:
        """Get statistics about embeddings."""
        query = db.query(models.Segment).filter(models.Segment.embedding.is_not(None))
        
        if ticker:
            query = query.filter_by(ticker=ticker.upper())
        if quarter:
            query = query.filter_by(quarter=quarter)
        
        embedded_segments = query.all()
        total_segments = db.query(models.Segment)
        
        if ticker:
            total_segments = total_segments.filter_by(ticker=ticker.upper())
        if quarter:
            total_segments = total_segments.filter_by(quarter=quarter)
        
        total_count = total_segments.count()
        
        return {
            "total_segments": total_count,
            "embedded_segments": len(embedded_segments),
            "coverage_percentage": (len(embedded_segments) / total_count * 100) if total_count > 0 else 0,
            "embedding_dimension": self.embedding_dim
        }

# Global instance (lazy loading)
finbert_encoder = None

def get_finbert_encoder():
    """Get or create FinBERT encoder instance."""
    global finbert_encoder
    if finbert_encoder is None:
        finbert_encoder = FinBERTEncoder()
    return finbert_encoder
