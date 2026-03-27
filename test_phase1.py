#!/usr/bin/env python3
"""
Test script to verify Phase 1 components are working correctly.
"""

import sys
import os

def test_imports():
    """Test that all Phase 1 modules can be imported."""
    print("Testing imports...")
    
    try:
        from app.intelligence.sarvam_client import sarvam_client
        print("✓ Sarvam client imported")
    except Exception as e:
        print(f"✗ Sarvam client import failed: {e}")
        return False
    
    try:
        from app.intelligence.entity_extractor import entity_extractor
        print("✓ Entity extractor imported")
    except Exception as e:
        print(f"✗ Entity extractor import failed: {e}")
        return False
    
    try:
        from app.intelligence.confidence_scorer import confidence_scorer
        print("✓ Confidence scorer imported")
    except Exception as e:
        print(f"✗ Confidence scorer import failed: {e}")
        return False
    
    try:
        from app.intelligence.embeddings import get_finbert_encoder
        print("✓ Embeddings module imported")
    except Exception as e:
        print(f"✗ Embeddings module import failed: {e}")
        return False
    
    try:
        from app.intelligence.report_generator import report_generator
        print("✓ Report generator imported")
    except Exception as e:
        print(f"✗ Report generator import failed: {e}")
        return False
    
    try:
        from app.tasks.celery_app import celery_app
        print("✓ Celery app imported")
    except Exception as e:
        print(f"✗ Celery app import failed: {e}")
        return False
    
    return True

def test_models():
    """Test that database models are correctly defined."""
    print("\nTesting database models...")
    
    try:
        from app import models
        
        # Check new models exist
        assert hasattr(models, 'EntityExtraction')
        assert hasattr(models, 'CallReport')
        print("✓ New models exist")
        
        # Check Segment model has new fields
        segment_columns = models.Segment.__table__.columns.keys()
        assert 'confidence_score' in segment_columns
        assert 'embedding' in segment_columns
        print("✓ Segment model has new fields")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_fastapi_endpoints():
    """Test that FastAPI app can be created with new endpoints."""
    print("\nTesting FastAPI endpoints...")
    
    try:
        from app.main import app
        
        # Check that intelligence endpoints exist
        route_paths = [route.path for route in app.routes]
        
        intelligence_routes = [
            "/intelligence/entities/{ticker}",
            "/intelligence/confidence/{ticker}",
            "/intelligence/embeddings/{ticker}",
            "/intelligence/full/{ticker}",
            "/intelligence/reports/{ticker}/{quarter}"
        ]
        
        for route in intelligence_routes:
            if route in route_paths:
                print(f"✓ {route} endpoint exists")
            else:
                print(f"✗ {route} endpoint missing")
                return False
        
        return True
    except Exception as e:
        print(f"✗ FastAPI test failed: {e}")
        return False

def test_environment():
    """Test that required environment variables are set."""
    print("\nTesting environment configuration...")
    
    required_vars = ['DATABASE_URL', 'SARVAM_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"✗ Missing environment variables: {missing_vars}")
        return False
    else:
        print("✓ Required environment variables set")
        return True

def main():
    """Run all tests."""
    print("=== Phase 1 Component Tests ===\n")
    
    tests = [
        ("imports", test_imports),
        ("models", test_models),
        ("endpoints", test_fastapi_endpoints),
        ("environment", test_environment)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append(result)
    
    print(f"\n=== Test Summary ===")
    success_count = sum(results)
    total_count = len(results)
    
    print(f"Passed: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All tests passed! Phase 1 is ready to use.")
        print("\nNext steps:")
        print("1. Start Redis: redis-server")
        print("2. Start Celery: celery -A app.tasks.celery_app worker --loglevel=info")
        print("3. Start FastAPI: uvicorn app.main:app --reload")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
