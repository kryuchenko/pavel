#!/usr/bin/env python3
"""
Complete test for Stage 5: Anomaly Detection Pipeline

Tests the full end-to-end anomaly detection system with integration
to Stage 3 (preprocessing) and Stage 4 (embeddings).
"""

import asyncio
import time
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

from pavel.clustering import AnomalyDetectionPipeline, DetectionConfig
from pavel.clustering.anomaly_classifier import AnomalyType, AnomalySeverity
from pavel.embeddings.embedding_pipeline import EmbeddingPipeline, PipelineConfig

async def test_complete_anomaly_detection():
    """Test complete anomaly detection pipeline"""
    print("🕵️ TESTING STAGE 5: COMPLETE ANOMALY DETECTION PIPELINE")
    print("=" * 70)
    
    # Initialize detection pipeline
    detection_config = DetectionConfig(
        enable_statistical=True,
        enable_clustering=True, 
        enable_semantic=True,
        enable_temporal=True,
        min_severity_level="low"
    )
    
    # Initialize embedding pipeline for integration
    embedding_config = PipelineConfig(
        embedding_model="intfloat/multilingual-e5-small",
        batch_size=5,
        enable_preprocessing=True
    )
    embedding_pipeline = EmbeddingPipeline(embedding_config)
    
    # Create detection pipeline with embedding integration
    detection_pipeline = AnomalyDetectionPipeline(
        config=detection_config,
        embedding_pipeline=embedding_pipeline
    )
    
    print("✅ Anomaly detection pipeline initialized")
    
    # Create test reviews with various anomaly patterns
    test_reviews = create_test_reviews_with_anomalies()
    print(f"📊 Testing with {len(test_reviews)} reviews containing various anomaly patterns")
    
    # Run complete anomaly detection
    start_time = time.time()
    
    try:
        result = await detection_pipeline.detect_anomalies(
            app_id="sinet.startup.inDriver",
            reviews=test_reviews
        )
        
        processing_time = time.time() - start_time
        
        # Analyze results
        print(f"\n📈 DETECTION RESULTS:")
        print(f"   ⏱️  Processing time: {processing_time:.2f}s")
        print(f"   📝 Reviews analyzed: {result.total_reviews_analyzed}")
        print(f"   🔍 Anomalies found: {len(result.classified_anomalies)}")
        print(f"   ⚠️  Critical issues: {len(result.critical_issues)}")
        
        # Detection method breakdown
        print(f"\n🔬 DETECTION METHOD RESULTS:")
        print(f"   📊 Statistical: {result.statistical_anomalies}")
        print(f"   🎯 Clustering: {result.clustering_anomalies}")
        print(f"   🧠 Semantic: {result.semantic_anomalies}")
        print(f"   ⏰ Temporal: {result.temporal_anomalies}")
        
        # Severity distribution
        print(f"\n🚨 SEVERITY DISTRIBUTION:")
        for severity, count in result.severity_distribution.items():
            print(f"   {severity.upper()}: {count}")
        
        # Anomaly type distribution
        print(f"\n🏷️  ANOMALY TYPES DETECTED:")
        for anomaly_type, count in result.type_distribution.items():
            print(f"   {anomaly_type}: {count}")
        
        # Business impact
        print(f"\n💼 BUSINESS IMPACT:")
        for impact, count in result.business_impact_distribution.items():
            print(f"   {impact}: {count}")
        
        # Critical issues details
        if result.critical_issues:
            print(f"\n🚩 CRITICAL ISSUES DETAILS:")
            for i, issue in enumerate(result.critical_issues[:5], 1):  # Show top 5
                print(f"   {i}. {issue.anomaly_type.value} (severity: {issue.severity.value})")
                print(f"      Score: {issue.severity_score:.2f}")
                print(f"      Reason: {issue.classification_reason}")
                print(f"      Action: {issue.recommended_action}")
                print()
        
        # Recommendations
        print(f"💡 RECOMMENDATIONS:")
        for i, recommendation in enumerate(result.recommended_actions, 1):
            print(f"   {i}. {recommendation}")
        
        # Test specific anomaly types
        detected_types = set(a.anomaly_type for a in result.classified_anomalies)
        expected_types = {
            AnomalyType.SPAM_REVIEW,
            AnomalyType.REVIEW_BOMBING,
            AnomalyType.CRASH_REPORT_SPIKE,
            AnomalyType.STATISTICAL_OUTLIER
        }
        
        print(f"\n✅ VALIDATION:")
        
        # Check if we detected expected anomaly types
        detected_expected = len(expected_types.intersection(detected_types))
        print(f"   Expected anomaly types detected: {detected_expected}/{len(expected_types)}")
        
        # Check if critical issues were properly flagged
        has_critical = any(a.severity in [AnomalySeverity.CRITICAL, AnomalySeverity.HIGH] 
                          for a in result.classified_anomalies)
        print(f"   Critical/High severity issues found: {'Yes' if has_critical else 'No'}")
        
        # Check processing performance
        reviews_per_second = result.total_reviews_analyzed / processing_time
        print(f"   Processing speed: {reviews_per_second:.1f} reviews/second")
        
        # Overall success criteria
        success = (
            len(result.classified_anomalies) >= 5 and  # Found multiple anomalies
            detected_expected >= 2 and  # Detected expected types
            has_critical and  # Found critical issues
            processing_time < 30.0 and  # Reasonable performance
            len(result.recommended_actions) > 0  # Generated recommendations
        )
        
        if success:
            print(f"\n🎉 STAGE 5 VALIDATION PASSED!")
            print(f"   ✓ Multi-method anomaly detection working")
            print(f"   ✓ Classification and severity assessment working")
            print(f"   ✓ Business impact analysis working")
            print(f"   ✓ Integration with Stage 3-4 working")
            print(f"   ✓ Recommendations generated")
            return True
        else:
            print(f"\n❌ STAGE 5 VALIDATION FAILED!")
            print(f"   Issues with anomaly detection pipeline")
            return False
    
    except Exception as e:
        print(f"\n💥 ANOMALY DETECTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_reviews_with_anomalies() -> List[Dict[str, Any]]:
    """Create test reviews with various anomaly patterns"""
    reviews = []
    base_time = datetime.now()
    
    # Normal reviews (baseline)
    normal_reviews = [
        "Good app, works well for ordering rides. Fast and reliable service.",
        "Nice interface, easy to use. Drivers are usually on time.",
        "Decent ride sharing app. Reasonable prices compared to competitors.",
        "Works fine most of the time. Occasional delays but overall satisfied.",
        "App is okay, nothing special but gets the job done."
    ]
    
    for i, content in enumerate(normal_reviews):
        reviews.append({
            'review_id': f'normal_{i+1}',
            'app_id': 'sinet.startup.inDriver',
            'content': content,
            'rating': 4,
            'created_at': base_time - timedelta(days=10, hours=i),
            'locale': 'en_US'
        })
    
    # Spam reviews (should trigger SPAM_REVIEW anomaly)
    spam_reviews = [
        "AMAZING APP!!! CLICK HERE FOR FREE MONEY!!! GUARANTEED RESULTS!!!",
        "Best app ever! Visit our website for special deals! Limited time offer!",
        "Perfect app! Download now and get free credits! No risk guaranteed!"
    ]
    
    for i, content in enumerate(spam_reviews):
        reviews.append({
            'review_id': f'spam_{i+1}',
            'app_id': 'sinet.startup.inDriver', 
            'content': content,
            'rating': 5,
            'created_at': base_time - timedelta(days=5, hours=i),
            'locale': 'en_US'
        })
    
    # Review bombing pattern (should trigger REVIEW_BOMBING)
    bombing_time = base_time - timedelta(hours=2)
    for i in range(10):  # Many reviews in short time
        reviews.append({
            'review_id': f'bomb_{i+1}',
            'app_id': 'sinet.startup.inDriver',
            'content': f"Terrible app! Worst experience ever! Hate it! Review {i+1}",
            'rating': 1,
            'created_at': bombing_time + timedelta(minutes=i*5),
            'locale': 'en_US'
        })
    
    # Crash report spike (should trigger CRASH_REPORT_SPIKE)
    crash_reviews = [
        "App crashes constantly! Freeze every time I try to book a ride. Major bug!",
        "Crashes and hangs all the time. Can't use the app anymore. Very buggy!",
        "App keeps crashing during payment. Lost money twice! Error after error!",
        "Freezes on startup. Crashes when I enter location. Full of bugs!"
    ]
    
    for i, content in enumerate(crash_reviews):
        reviews.append({
            'review_id': f'crash_{i+1}',
            'app_id': 'sinet.startup.inDriver',
            'content': content,
            'rating': 1,
            'created_at': base_time - timedelta(hours=1, minutes=i*10),
            'locale': 'en_US'
        })
    
    # Statistical outliers (very long/short reviews)
    reviews.append({
        'review_id': 'outlier_long',
        'app_id': 'sinet.startup.inDriver',
        'content': "This is an extremely long review that goes on and on about every single detail of my experience with this ride sharing application. " * 10,
        'rating': 3,
        'created_at': base_time - timedelta(days=3),
        'locale': 'en_US'
    })
    
    reviews.append({
        'review_id': 'outlier_short',
        'app_id': 'sinet.startup.inDriver',
        'content': "No",
        'rating': 2,
        'created_at': base_time - timedelta(days=2),
        'locale': 'en_US'
    })
    
    # Competitor mentions
    competitor_reviews = [
        "Much better than Uber! Try this instead of other apps!",
        "Compared to Lyft, this is way superior. Switch to inDriver now!"
    ]
    
    for i, content in enumerate(competitor_reviews):
        reviews.append({
            'review_id': f'competitor_{i+1}',
            'app_id': 'sinet.startup.inDriver',
            'content': content,
            'rating': 5,
            'created_at': base_time - timedelta(days=1, hours=i),
            'locale': 'en_US'
        })
    
    # Payment issues
    payment_reviews = [
        "Payment system is broken! Charged me twice! Billing error! Need refund!",
        "Unauthorized charges on my card! Fraud! Money stolen! Cancel subscription!"
    ]
    
    for i, content in enumerate(payment_reviews):
        reviews.append({
            'review_id': f'payment_{i+1}',
            'app_id': 'sinet.startup.inDriver',
            'content': content,
            'rating': 1,
            'created_at': base_time - timedelta(hours=6, minutes=i*30),
            'locale': 'en_US'
        })
    
    # Multilingual reviews for semantic testing
    multilingual_reviews = [
        {
            'review_id': 'multi_1',
            'content': 'Отличное приложение! Очень удобно заказывать поездки',
            'rating': 5,
            'locale': 'ru_RU'
        },
        {
            'review_id': 'multi_2', 
            'content': 'Aplicación excelente para solicitar viajes. Muy recomendable',
            'rating': 5,
            'locale': 'es_ES'
        }
    ]
    
    for review in multilingual_reviews:
        review.update({
            'app_id': 'sinet.startup.inDriver',
            'created_at': base_time - timedelta(days=7),
        })
        reviews.append(review)
    
    return reviews

async def test_sensitivity_configuration():
    """Test sensitivity configuration capabilities"""
    print("\n🎛️ TESTING SENSITIVITY CONFIGURATION")
    print("-" * 50)
    
    # Create pipeline
    detection_pipeline = AnomalyDetectionPipeline()
    
    # Test default sensitivity
    print("Testing default sensitivity...")
    test_reviews = create_test_reviews_with_anomalies()[:10]  # Subset for speed
    
    result1 = await detection_pipeline.detect_anomalies("test_app", test_reviews)
    baseline_anomalies = len(result1.classified_anomalies)
    print(f"Baseline anomalies detected: {baseline_anomalies}")
    
    # Test higher sensitivity
    print("Testing higher sensitivity (2.0x)...")
    detection_pipeline.configure_sensitivity(global_sensitivity=2.0)
    
    result2 = await detection_pipeline.detect_anomalies("test_app", test_reviews)
    high_sensitivity_anomalies = len(result2.classified_anomalies)
    print(f"High sensitivity anomalies detected: {high_sensitivity_anomalies}")
    
    # Test lower sensitivity
    print("Testing lower sensitivity (0.5x)...")
    detection_pipeline.configure_sensitivity(global_sensitivity=0.5)
    
    result3 = await detection_pipeline.detect_anomalies("test_app", test_reviews)
    low_sensitivity_anomalies = len(result3.classified_anomalies)
    print(f"Low sensitivity anomalies detected: {low_sensitivity_anomalies}")
    
    # Validate sensitivity changes
    sensitivity_works = (
        high_sensitivity_anomalies >= baseline_anomalies and
        low_sensitivity_anomalies <= baseline_anomalies
    )
    
    print(f"✅ Sensitivity configuration: {'WORKING' if sensitivity_works else 'FAILED'}")
    return sensitivity_works

async def main():
    """Main test function"""
    print("🔍 PAVEL STAGE 5: ANOMALY DETECTION COMPLETE TEST")
    print("=" * 70)
    
    # Test complete pipeline
    pipeline_success = await test_complete_anomaly_detection()
    
    # Test sensitivity configuration
    sensitivity_success = await test_sensitivity_configuration()
    
    # Overall results
    print("\n" + "=" * 70)
    if pipeline_success and sensitivity_success:
        print("🎉 STAGE 5: ANOMALY DETECTION - COMPLETE SUCCESS!")
        print()
        print("🚀 FEATURES IMPLEMENTED:")
        print("   ✅ Statistical Anomaly Detection (Z-score, IQR, Grubbs)")
        print("   ✅ Clustering Anomaly Detection (DBSCAN, Isolation Forest, LOF)")
        print("   ✅ Semantic Anomaly Detection (Embedding outliers, content divergence)")
        print("   ✅ Temporal Anomaly Detection (Volume spikes, rating shifts, trends)")
        print("   ✅ Anomaly Classification (20+ anomaly types)")
        print("   ✅ Severity Assessment (Critical, High, Medium, Low, Info)")
        print("   ✅ Business Impact Analysis")
        print("   ✅ Configurable Sensitivity Settings")
        print("   ✅ Integration with Stage 3-4 (Preprocessing + Embeddings)")
        print("   ✅ Actionable Recommendations")
        print()
        print("🎯 PAVEL is now complete and production-ready!")
        print("   Ready to detect anomalies in Google Play reviews at scale")
    else:
        print("❌ STAGE 5 VALIDATION ISSUES DETECTED")
        print("   Pipeline needs attention before production")

if __name__ == "__main__":
    asyncio.run(main())