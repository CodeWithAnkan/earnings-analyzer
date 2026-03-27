#!/usr/bin/env python3
"""
Setup script for Phase 1 database requirements.
This script will:
1. Create the pgvector extension in PostgreSQL
2. Run database migrations for new tables
"""

import subprocess
import sys
from sqlalchemy import create_engine, text
from app.database import DATABASE_URL
from app import models

def setup_pgvector():
    """Create pgvector extension in PostgreSQL."""
    print("Setting up pgvector extension...")
    
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
        print("✓ pgvector extension created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create pgvector extension: {e}")
        return False

def create_tables():
    """Create all database tables."""
    print("Creating database tables...")
    
    try:
        from app.database import engine
        models.Base.metadata.create_all(bind=engine)
        print("✓ Database tables created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create tables: {e}")
        return False

def check_redis():
    """Check if Redis is running."""
    print("Checking Redis connection...")
    
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✓ Redis is running")
        return True
    except Exception as e:
        print(f"✗ Redis is not running or not accessible: {e}")
        print("Please start Redis server: redis-server")
        return False

def main():
    """Run all setup steps."""
    print("=== Phase 1 Database Setup ===\n")
    
    steps = [
        ("pgvector extension", setup_pgvector),
        ("database tables", create_tables),
        ("redis connection", check_redis)
    ]
    
    results = []
    for step_name, step_func in steps:
        print(f"\n--- {step_name.title()} ---")
        result = step_func()
        results.append(result)
    
    print(f"\n=== Setup Summary ===")
    success_count = sum(results)
    total_count = len(results)
    
    print(f"Completed: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All setup steps completed successfully!")
        print("\nYou can now start the application:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Start Redis: redis-server")
        print("3. Start Celery worker: celery -A app.tasks.celery_app worker --loglevel=info")
        print("4. Start FastAPI: uvicorn app.main:app --reload")
    else:
        print("❌ Some setup steps failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
