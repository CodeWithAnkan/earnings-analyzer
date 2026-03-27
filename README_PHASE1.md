# Phase 1: Sarvam Intelligence Layer

This document describes the Phase 1 implementation of the Sarvam Intelligence Layer for earnings call analysis.

## Overview

Phase 1 adds intelligent analysis capabilities to the earnings analyzer:
- **Sarvam API Integration**: Entity extraction (guidance, risks, metrics)
- **Confidence Scoring**: Hedge-word analysis per speaker turn
- **FinBERT Embeddings**: Semantic similarity search using financial BERT
- **Background Processing**: Celery + Redis for async ML operations
- **Structured Reports**: JSON reports per earnings call

## New Features

### 1. Entity Extraction
Extracts structured entities from earnings call segments:
- **Guidance**: Forward-looking statements, forecasts, targets
- **Risks**: Risk factors, concerns, challenges
- **Metrics**: Specific numbers, percentages, financial figures

### 2. Confidence Scoring
Analyzes speaker confidence based on language patterns:
- Scores range from 0.0 (very uncertain) to 1.0 (very confident)
- Identifies hedge words and uncertainty phrases
- Per-speaker confidence tracking

### 3. Semantic Search
FinBERT-powered embeddings enable:
- Find similar segments across all earnings calls
- Vector similarity search in PostgreSQL
- 768-dimensional embeddings for semantic understanding

### 4. Background Processing
All ML operations run asynchronously:
- Celery workers process heavy computations
- Redis for task queue and results storage
- Progress tracking for long-running tasks

## API Endpoints

### Intelligence Processing
```
POST /intelligence/entities/{ticker}           # Extract entities
POST /intelligence/confidence/{ticker}         # Score confidence
POST /intelligence/embeddings/{ticker}          # Generate embeddings
POST /intelligence/full/{ticker}                # Run full pipeline
```

### Data Retrieval
```
GET /intelligence/entities/{ticker}?type=guidance|risks|metrics
GET /intelligence/confidence/{ticker}          # Confidence stats
GET /intelligence/low-confidence/{ticker}       # Low confidence segments
GET /intelligence/similar?query=...             # Semantic search
GET /intelligence/embeddings/stats/{ticker}     # Embedding stats
```

### Reports
```
POST /intelligence/reports/{ticker}/{quarter}    # Generate report
GET /intelligence/reports/{ticker}/{quarter}     # Get saved report
```

### Task Management
```
GET /intelligence/tasks/{task_id}                 # Task status
```

## Setup Instructions

### 1. Prerequisites
```bash
# PostgreSQL with pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

# Redis server
redis-server

# Python 3.8+
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create/update `.env` file:
```env
# Database
DATABASE_URL=postgresql://user:password@localhost/earnings_db

# APIs
ALPHAVANTAGE_KEY=your_alpha_vantage_key
SARVAM_API_KEY=your_sarvam_api_key

# Redis
REDIS_URL=redis://localhost:6379/0
```

### 4. Database Setup
```bash
python setup_database.py
```

### 5. Start Services
```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start Celery worker
celery -A app.tasks.celery_app worker --loglevel=info

# Terminal 3: Start FastAPI server
uvicorn app.main:app --reload
```

## Usage Examples

### 1. Ingest and Process Earnings Calls
```bash
# Ingest transcripts
curl -X POST "http://localhost:8000/ingest/NVDA"

# Run full intelligence pipeline
curl -X POST "http://localhost:8000/intelligence/full/NVDA"

# Check task status
curl "http://localhost:8000/intelligence/tasks/{task_id}"
```

### 2. Query Results
```bash
# Get extracted guidance
curl "http://localhost:8000/intelligence/entities/NVDA?entity_type=guidance"

# Get confidence statistics
curl "http://localhost:8000/intelligence/confidence/NVDA"

# Find similar segments
curl "http://localhost:8000/intelligence/similar?query=revenue%20growth&ticker=NVDA"

# Get comprehensive report
curl "http://localhost:8000/intelligence/reports/NVDA/2025Q3"
```

## Database Schema

### New Tables
- `entity_extractions`: Stores extracted entities per segment
- `call_reports`: Structured JSON reports per earnings call

### Modified Tables
- `segments`: Added `confidence_score` and `embedding` columns

### Vector Storage
- Uses PostgreSQL pgvector extension
- 768-dimensional vectors for FinBERT embeddings
- Supports cosine similarity search

## Architecture

### Components
1. **Sarvam Client**: OpenAI-compatible API wrapper
2. **Entity Extractor**: Structured data extraction
3. **Confidence Scorer**: Language pattern analysis
4. **FinBERT Encoder**: Semantic embedding generation
5. **Report Generator**: Comprehensive call analysis
6. **Celery Tasks**: Background job processing

### Data Flow
```
Transcript → Segments → Entity Extraction → Confidence Scoring → Embeddings → Report
     ↓              ↓              ↓                ↓              ↓        ↓
  PostgreSQL   Background Jobs   Sarvam API      Sarvam API    FinBERT   JSON
```

## Performance Considerations

### Batch Processing
- Entity extraction: 50 segments per batch
- Embedding generation: 8 segments per batch
- Configurable batch sizes

### Caching
- FinBERT model loaded once per worker
- Redis for task result caching
- Database connection pooling

### Monitoring
- Task progress tracking
- Error handling and retry logic
- Performance metrics per operation

## Troubleshooting

### Common Issues
1. **pgvector extension not found**: Run `CREATE EXTENSION vector;` in PostgreSQL
2. **Redis connection failed**: Start Redis server with `redis-server`
3. **Sarvam API errors**: Check API key and base URL configuration
4. **FinBERT download timeout**: Model downloads on first use (~500MB)

### Debug Mode
Enable verbose logging:
```bash
celery -A app.tasks.celery_app worker --loglevel=debug
uvicorn app.main:app --log-level debug
```

## Next Steps (Phase 2)

Phase 1 establishes the ML foundation. Phase 2 will focus on:
- Drift detection algorithms
- Time-series analysis of confidence trends
- Advanced visualization dashboard
- Real-time processing capabilities

## Success Metrics

Phase 1 success criteria:
- ✅ Sarvam API extracting entities from 100% of segments
- ✅ Confidence scores generated for all speaker turns
- ✅ FinBERT embeddings stored in pgvector
- ✅ Background job system operational
- ✅ Structured JSON reports generated per call
