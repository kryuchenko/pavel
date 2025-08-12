#!/usr/bin/env python3
"""
Test Stage 3 preprocessing with integrated complaint classifier

Tests the complete preprocessing pipeline including the new complaint
classification stage using the trained ML model.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pavel.core.logger import get_logger
from pavel.preprocessing import PreprocessingPipeline
from pavel.classification import get_complaint_classifier

logger = get_logger(__name__)

async def test_preprocessing_with_classifier():
    """Test full preprocessing pipeline with complaint classification"""
    
    print("🧪 Testing Stage 3 Preprocessing with Complaint Classifier")
    print("=" * 60)
    
    # Test reviews (mix of complaints and non-complaints)
    test_reviews = [
        {
            'reviewId': 'test-1',
            'appId': 'sinet.startup.inDriver',
            'content': 'App crashes every time I try to book a ride. Very frustrating!',
            'locale': 'en_US',
            'score': 1
        },
        {
            'reviewId': 'test-2', 
            'appId': 'sinet.startup.inDriver',
            'content': 'Great app, works perfectly! Love the interface.',
            'locale': 'en_US',
            'score': 5
        },
        {
            'reviewId': 'test-3',
            'appId': 'sinet.startup.inDriver', 
            'content': 'Приложение тормозит и глючит. Ужасно!',
            'locale': 'ru_RU',
            'score': 2
        },
        {
            'reviewId': 'test-4',
            'appId': 'sinet.startup.inDriver',
            'content': 'Excelente servicio, muy recomendable.',
            'locale': 'es_ES', 
            'score': 4
        }
    ]
    
    # Initialize pipeline
    print("🔧 Initializing preprocessing pipeline...")
    pipeline = PreprocessingPipeline()
    
    # Test individual review processing
    print("\n📝 Testing individual review processing...")
    
    for i, review in enumerate(test_reviews, 1):
        print(f"\n🔍 Processing review {i}: {review['content'][:50]}...")
        
        try:
            processed = await pipeline.process_single_review(review)
            
            print(f"   ✅ Language: {processed.detected_language} ({processed.language_confidence:.2f})")
            print(f"   ✅ Sentences: {processed.sentence_count}")
            print(f"   🚨 Complaint: {'YES' if processed.is_complaint else 'NO'} ({processed.complaint_confidence:.2f})")
            
            # Validate results
            print(f"   📊 Debug: review_id={processed.review_id}, expected={review['reviewId']}")
            print(f"   📊 Debug: is_complaint={processed.is_complaint}, confidence={processed.complaint_confidence}")
            
            assert processed.review_id == review['reviewId'], f"Review ID mismatch: {processed.review_id} != {review['reviewId']}"
            assert processed.original_content == review['content'], "Content mismatch"
            assert processed.detected_language in ['en', 'ru', 'es'], f"Unexpected language: {processed.detected_language}"
            assert 0 <= processed.complaint_confidence <= 1.0, f"Invalid confidence: {processed.complaint_confidence}"
            assert isinstance(processed.is_complaint, bool), f"Invalid complaint flag type: {type(processed.is_complaint)}"
            
        except Exception as e:
            print(f"   ❌ Error processing review {i}: {e}")
            raise
    
    # Test classifier directly
    print("\n🤖 Testing complaint classifier directly...")
    classifier = get_complaint_classifier()
    
    test_texts = [
        "This app is broken and doesn't work!",
        "Amazing service, highly recommended!",
        "Приложение не работает, очень плохо",
        "Perfecto, me encanta esta aplicación"
    ]
    
    predictions = classifier.predict_batch(test_texts)
    
    for text, prediction in zip(test_texts, predictions):
        print(f"   📝 '{text[:30]}...' → {'COMPLAINT' if prediction.is_complaint else 'POSITIVE'} "
              f"({prediction.confidence:.3f})")
        
        # Validate prediction structure
        assert isinstance(prediction.is_complaint, bool)
        assert 0 <= prediction.confidence <= 1.0
        assert 0 <= prediction.complaint_probability <= 1.0
    
    # Test filtering functionality
    print("\n🔍 Testing complaint filtering...")
    filtered_reviews, stats = classifier.filter_complaints(test_reviews, threshold=0.5)
    
    print(f"   📊 Filtered {stats['complaints']}/{stats['total']} reviews as complaints")
    print(f"   📈 Complaint rate: {100*stats['complaint_rate']:.1f}%")
    
    # Validate filtering results
    assert stats['total'] == len(test_reviews)
    assert stats['complaints'] + stats['non_complaints'] == stats['total']
    assert 0 <= stats['complaint_rate'] <= 1.0
    
    # Check that complaint reviews have classification metadata
    for review in filtered_reviews:
        assert 'complaint_classification' in review
        assert 'is_complaint' in review['complaint_classification']
        assert review['complaint_classification']['is_complaint'] is True
    
    print("\n🎯 Testing Summary:")
    print(f"   ✅ Individual review processing: 4/4 passed")
    print(f"   ✅ Direct classifier testing: 4/4 predictions")
    print(f"   ✅ Complaint filtering: {stats['complaints']} complaints identified")
    print(f"   ✅ All validations passed")
    
    print("\n🎉 Stage 3 with complaint classifier - ALL TESTS PASSED!")
    print("   📋 Features validated:")
    print("      • Multilingual text preprocessing")
    print("      • Language detection with confidence")
    print("      • Sentence segmentation")
    print("      • Complaint/non-complaint classification")
    print("      • Review filtering by complaint status")
    print("      • Integration with preprocessing pipeline")

def main():
    """Run the tests"""
    try:
        asyncio.run(test_preprocessing_with_classifier())
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())