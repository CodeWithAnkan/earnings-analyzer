#!/usr/bin/env python3
"""
Example usage of Phase 1 intelligence features.
This script demonstrates how to use the new intelligence components.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def example_full_pipeline():
    """Example: Run full intelligence pipeline for a ticker."""
    print("=== Full Intelligence Pipeline Example ===\n")
    
    ticker = "NVDA"
    
    # Step 1: Ingest transcripts (if not already done)
    print("1. Ingesting transcripts...")
    response = requests.post(f"{BASE_URL}/ingest/{ticker}")
    if response.status_code == 200:
        print(f"✓ Ingested: {response.json()}")
    else:
        print(f"⚠ Ingestion may have failed or data exists: {response.status_code}")
    
    # Step 2: Run full intelligence pipeline
    print("\n2. Starting full intelligence pipeline...")
    response = requests.post(f"{BASE_URL}/intelligence/full/{ticker}")
    if response.status_code == 200:
        task_data = response.json()
        print(f"✓ Tasks queued: {task_data}")
        task_ids = task_data["task_ids"]
        
        # Step 3: Monitor progress
        print("\n3. Monitoring task progress...")
        for task_info in task_ids:
            task_id = task_info["task_id"]
            quarter = task_info["quarter"]
            print(f"\n   Monitoring {quarter} (Task: {task_id})")
            
            while True:
                status_response = requests.get(f"{BASE_URL}/intelligence/tasks/{task_id}")
                if status_response.status_code == 200:
                    status = status_response.json()
                    state = status.get("state", "UNKNOWN")
                    
                    if state == "SUCCESS":
                        print(f"   ✓ {quarter} completed successfully")
                        break
                    elif state == "FAILURE":
                        print(f"   ✗ {quarter} failed: {status.get('error', 'Unknown error')}")
                        break
                    elif state == "PROGRESS":
                        progress = status.get("progress", 0)
                        processed = status.get("processed", 0)
                        total = status.get("total", 0)
                        print(f"   ⏳ {quarter}: {progress:.1f}% ({processed}/{total})")
                    
                    time.sleep(2)
                else:
                    print(f"   ⚠ Could not get status for {task_id}")
                    break
    else:
        print(f"✗ Failed to start pipeline: {response.status_code}")
        return
    
    # Step 4: View results
    print("\n4. Viewing results...")
    
    # Get entities
    for entity_type in ["guidance", "risks", "metrics"]:
        response = requests.get(f"{BASE_URL}/intelligence/entities/{ticker}?entity_type={entity_type}")
        if response.status_code == 200:
            data = response.json()
            entities = data.get("entities", [])
            print(f"\n   {entity_type.title()}: {len(entities)} items")
            for entity in entities[:3]:  # Show first 3
                print(f"     - {entity['entity_value'][:80]}...")
    
    # Get confidence stats
    response = requests.get(f"{BASE_URL}/intelligence/confidence/{ticker}")
    if response.status_code == 200:
        stats = response.json()["confidence_stats"]
        print(f"\n   Confidence Stats:")
        print(f"     - Average: {stats['avg_confidence']:.2f}")
        print(f"     - Range: {stats['min_confidence']:.2f} - {stats['max_confidence']:.2f}")
        print(f"     - Segments scored: {stats['count']}")
    
    # Get embedding stats
    response = requests.get(f"{BASE_URL}/intelligence/embeddings/stats/{ticker}")
    if response.status_code == 200:
        stats = response.json()["embedding_stats"]
        print(f"\n   Embedding Stats:")
        print(f"     - Coverage: {stats['coverage_percentage']:.1f}%")
        print(f"     - Dimension: {stats['embedding_dimension']}")
    
    # Generate and view report
    print("\n5. Generating reports...")
    response = requests.post(f"{BASE_URL}/intelligence/reports/{ticker}/2025Q3")
    if response.status_code == 200:
        report_data = response.json()
        print(f"✓ Report generated: {report_data}")
        
        # Get the full report
        response = requests.get(f"{BASE_URL}/intelligence/reports/{ticker}/2025Q3")
        if response.status_code == 200:
            report = response.json()
            print(f"\n   Report Summary:")
            print(f"     - Total segments: {report['summary']['total_segments']}")
            print(f"     - Key insights: {len(report['key_insights'])}")
            for insight in report['key_insights']:
                print(f"       • {insight}")

def example_semantic_search():
    """Example: Semantic search using embeddings."""
    print("\n=== Semantic Search Example ===\n")
    
    queries = [
        "revenue growth forecast",
        "market challenges",
        "cost reduction initiatives"
    ]
    
    for query in queries:
        print(f"Searching: '{query}'")
        response = requests.get(f"{BASE_URL}/intelligence/similar?query={query}&limit=3")
        if response.status_code == 200:
            results = response.json()["similar_segments"]
            print(f"Found {len(results)} similar segments:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Similarity: {result['similarity']:.3f}")
                print(f"     Speaker: {result['speaker']} ({result['role']})")
                print(f"     Preview: {result['text_preview'][:100]}...")
                print()

def example_low_confidence_analysis():
    """Example: Find low confidence segments."""
    print("\n=== Low Confidence Analysis ===\n")
    
    response = requests.get(f"{BASE_URL}/intelligence/low-confidence/NVDA?threshold=0.3")
    if response.status_code == 200:
        data = response.json()
        segments = data["segments"]
        print(f"Found {len(segments)} low confidence segments (< 0.3):")
        
        for i, segment in enumerate(segments[:5], 1):
            print(f"\n{i}. Confidence: {segment['confidence_score']:.2f}")
            print(f"   Speaker: {segment['speaker']} ({segment['role']})")
            print(f"   Preview: {segment['text_preview'][:150]}...")

def main():
    """Run all examples."""
    print("Phase 1 Intelligence Features Demo")
    print("===================================\n")
    
    try:
        # Check if server is running
        response = requests.get(f"{BASE_URL}/")
        if response.status_code != 200:
            print("❌ Server not running. Please start with:")
            print("   uvicorn app.main:app --reload")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Please start with:")
        print("   uvicorn app.main:app --reload")
        return
    
    print("✓ Server is running\n")
    
    # Run examples
    example_full_pipeline()
    example_semantic_search()
    example_low_confidence_analysis()
    
    print("\n=== Demo Complete ===")
    print("Try these API endpoints directly:")
    print("- GET /intelligence/entities/NVDA?entity_type=guidance")
    print("- GET /intelligence/confidence/NVDA")
    print("- GET /intelligence/similar?query=artificial%20intelligence")
    print("- POST /intelligence/reports/NVDA/2025Q3")

if __name__ == "__main__":
    main()
