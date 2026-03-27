#!/usr/bin/env python3
"""
Setup script for Phase 1.
1. Creates pgvector extension in PostgreSQL
2. Creates all database tables
3. Checks Redis connection
"""

import sys
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def setup_pgvector():
    print("Checking pgvector extension...")
    try:
        from app.database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector';"))
            if result.fetchone():
                print("✓ pgvector extension is active")
                return True
            else:
                print("✗ pgvector not found — run this manually:")
                print("  docker exec -it earnings-analyzer-db-1 psql -U postgres -d earnings_db -c \"CREATE EXTENSION vector;\"")
                return False
    except Exception as e:
        print(f"✗ Could not verify pgvector: {e}")
        return False

def create_tables():
    print("Creating database tables...")
    try:
        from app.database import engine
        from app import models
        models.Base.metadata.create_all(bind=engine)
        print("✓ Tables created")
        return True
    except Exception as e:
        print(f"✗ Table creation failed: {e}")
        return False


def check_redis():
    print("Checking Redis...")
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, db=0)
        r.ping()
        print("✓ Redis is running")
        return True
    except Exception as e:
        print(f"✗ Redis not reachable: {e}")
        print("  Start Redis with: redis-server")
        return False


def main():
    print("=== Phase 1 Setup ===\n")

    steps = [
        ("pgvector", setup_pgvector),
        ("tables", create_tables),
        ("redis", check_redis),
    ]

    results = [fn() for _, fn in steps]
    passed = sum(results)

    print(f"\n{'✅ All good!' if passed == len(steps) else '❌ Some steps failed.'} ({passed}/{len(steps)})")

    if passed == len(steps):
        print("\nNext steps:")
        print("  redis-server")
        print("  celery -A app.tasks.celery_app worker --loglevel=info")
        print("  uvicorn app.main:app --reload")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()