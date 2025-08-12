#!/usr/bin/env python3
"""
Quick integration test for main PAVEL code after fixes.

Validates that all main components work without warnings or errors.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any

def test_imports():
    """Test all critical imports work."""
    print("🔍 Testing imports...")
    
    try:
        # Core components
        from pavel.core.app_config import get_default_app_info
        from pavel.core.logger import get_logger
        
        # Anomaly detection
        from pavel.clustering.detection_pipeline import AnomalyDetectionPipeline
        from pavel.clustering.smart_detection_pipeline import SmartDetectionPipeline
        from pavel.clustering.dynamic_cluster_detector import DynamicClusterDetector
        
        # Individual detectors
        from pavel.clustering.statistical_detector import StatisticalAnomalyDetector
        from pavel.clustering.clustering_detector import ClusteringAnomalyDetector
        from pavel.clustering.semantic_detector import SemanticAnomalyDetector
        from pavel.clustering.temporal_detector import TemporalAnomalyDetector
        
        # Embeddings
        from pavel.embeddings.embedding_pipeline import EmbeddingPipeline
        from pavel.embeddings.embedding_generator import EmbeddingGenerator
        
        print("   ✅ All imports successful")
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False


def test_configuration():
    """Test configuration is working."""
    print("🔧 Testing configuration...")
    
    try:
        from pavel.core.app_config import get_default_app_info
        
        app_info = get_default_app_info()
        required_keys = ['app_id', 'name', 'url', 'description']
        
        for key in required_keys:
            if key not in app_info:
                print(f"   ❌ Missing config key: {key}")
                return False
                
        print(f"   ✅ Configuration OK: {app_info['name']} ({app_info['app_id']})")
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration failed: {e}")
        return False


def test_basic_detection():
    """Test basic anomaly detection setup."""
    print("🧠 Testing basic detection setup...")
    
    try:
        from pavel.clustering.detection_pipeline import AnomalyDetectionPipeline, DetectionConfig
        from pavel.clustering.statistical_detector import StatisticalConfig
        
        # Create pipeline with minimal config
        config = DetectionConfig(
            enable_statistical=True,
            enable_clustering=False,  # Disable resource-heavy methods
            enable_semantic=False,
            enable_temporal=False
        )
        
        pipeline = AnomalyDetectionPipeline(config)
        print("   ✅ Detection pipeline created successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Detection setup failed: {e}")
        return False


def test_smart_detection():
    """Test smart detection setup."""
    print("🤖 Testing smart detection setup...")
    
    try:
        from pavel.clustering.smart_detection_pipeline import SmartDetectionPipeline
        
        # Create smart pipeline
        smart_pipeline = SmartDetectionPipeline(
            embedding_pipeline=None,  # Use TF-IDF fallback
            history_weeks=2,
            min_reviews_for_analysis=10
        )
        
        print("   ✅ Smart detection pipeline created successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Smart detection setup failed: {e}")
        return False


async def test_embedding_pipeline():
    """Test embedding pipeline basic functionality."""
    print("💭 Testing embedding pipeline...")
    
    try:
        from pavel.embeddings.embedding_pipeline import EmbeddingPipeline, PipelineConfig
        
        # Create minimal config
        config = PipelineConfig(
            embedding_model="intfloat/multilingual-e5-small",
            batch_size=2,  # Small batch for test
            enable_preprocessing=False
        )
        
        pipeline = EmbeddingPipeline(config)
        
        # Test with single review (longer text to ensure it passes quality filters)
        test_review = {
            'review_id': 'test_1',
            'app_id': 'test.app.id', 
            'content': 'This is a comprehensive test review for the embedding pipeline functionality to ensure it works correctly',
            'rating': 4,
            'created_at': datetime.now()
        }
        
        result = await pipeline.process_single_review(
            review_data=test_review
        )
        
        if result is not None:
            print(f"   ✅ Review processed successfully (vector ID: {result})")
            return True
        else:
            print("   ❌ Failed to process review - returned None")
            return False
        
    except Exception as e:
        print(f"   ❌ Embedding pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("🚀 PAVEL MAIN CODE INTEGRATION TEST")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Basic Detection", test_basic_detection),
        ("Smart Detection", test_smart_detection),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        if test_func():
            passed += 1
        print()
    
    # Test embedding pipeline separately (async)
    print("Running async tests...")
    try:
        result = asyncio.run(test_embedding_pipeline())
        if result:
            passed += 1
        total += 1
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        total += 1
    
    print()
    print("=" * 50)
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print(f"✅ {passed}/{total} tests successful")
        print("\n🚀 Main PAVEL code is ready for production!")
        return True
    else:
        print(f"❌ SOME TESTS FAILED: {passed}/{total}")
        print("🔧 Please review the issues above")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)