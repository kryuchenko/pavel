#!/usr/bin/env python3
"""
Real-world language detection test using actual Google Play reviews.

Fetches 50 reviews from different locales and tests language detection accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google_play_scraper import reviews, Sort
from pavel.preprocessing.language_detector import LanguageDetector
from pavel.core.logger import get_logger
from collections import Counter

logger = get_logger(__name__)

def fetch_real_reviews(app_id: str = "com.whatsapp", sample_size: int = 50):
    """Fetch real reviews from Google Play with known locales"""
    
    all_reviews = []
    
    # Define locales with expected languages
    locale_configs = [
        ("en", "US", "en", 10),  # English
        ("ru", "RU", "ru", 10),  # Russian
        ("es", "ES", "es", 10),  # Spanish
        ("pt", "BR", "pt", 10),  # Portuguese
        ("id", "ID", "id", 10),  # Indonesian
    ]
    
    logger.info(f"Fetching {sample_size} real reviews from Google Play...")
    
    for lang, country, expected_lang, count in locale_configs:
        try:
            logger.info(f"Fetching {count} reviews from {lang}_{country}...")
            result, _ = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=count
            )
            
            for review in result[:count]:
                all_reviews.append({
                    'content': review.get('content', ''),
                    'locale': f"{lang}_{country}",
                    'expected_language': expected_lang,
                    'score': review.get('score'),
                    'at': review.get('at')
                })
                
            logger.info(f"✓ Got {len(result[:count])} reviews from {lang}_{country}")
            
        except Exception as e:
            logger.warning(f"Failed to fetch from {lang}_{country}: {e}")
            
    return all_reviews

def test_language_detection(reviews_data):
    """Test language detection on real reviews"""
    
    detector = LanguageDetector(min_confidence=0.5)
    
    results = []
    correct = 0
    total = 0
    
    logger.info("\n" + "="*80)
    logger.info("LANGUAGE DETECTION TEST ON REAL REVIEWS")
    logger.info("="*80)
    
    for i, review_data in enumerate(reviews_data, 1):
        content = review_data['content']
        expected = review_data['expected_language']
        locale = review_data['locale']
        
        if not content or len(content.strip()) < 10:
            continue
            
        # Detect language
        detection = detector.detect(content, locale)
        
        # Check if correct
        is_correct = detection.language == expected
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            'review': content[:100] + "..." if len(content) > 100 else content,
            'locale': locale,
            'expected': expected,
            'detected': detection.language,
            'confidence': detection.confidence,
            'method': detection.method,
            'correct': is_correct
        })
        
        # Log detailed results for incorrect detections
        if not is_correct:
            logger.warning(f"\n✗ Incorrect detection #{i}:")
            logger.warning(f"  Review: {content[:100]}...")
            logger.warning(f"  Expected: {expected}, Got: {detection.language} ({detection.confidence:.2f})")
            logger.warning(f"  Method: {detection.method}")
            if detection.alternatives:
                logger.warning(f"  Alternatives: {detection.alternatives[:3]}")
    
    # Calculate statistics
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Group by expected language
    by_language = {}
    for result in results:
        lang = result['expected']
        if lang not in by_language:
            by_language[lang] = {'correct': 0, 'total': 0}
        by_language[lang]['total'] += 1
        if result['correct']:
            by_language[lang]['correct'] += 1
    
    # Method statistics
    method_stats = Counter(r['method'] for r in results)
    
    # Confidence statistics
    confidence_stats = {
        'high': len([r for r in results if r['confidence'] >= 0.8]),
        'medium': len([r for r in results if 0.5 <= r['confidence'] < 0.8]),
        'low': len([r for r in results if r['confidence'] < 0.5])
    }
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("DETECTION RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"\n📊 Overall Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    logger.info("\n📈 Accuracy by Language:")
    for lang, stats in sorted(by_language.items()):
        lang_accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        logger.info(f"  {lang}: {stats['correct']}/{stats['total']} ({lang_accuracy:.1f}%)")
    
    logger.info("\n🔧 Detection Methods Used:")
    for method, count in method_stats.most_common():
        logger.info(f"  {method}: {count} ({count/total*100:.1f}%)")
    
    logger.info("\n💯 Confidence Distribution:")
    logger.info(f"  High (≥0.8): {confidence_stats['high']} ({confidence_stats['high']/total*100:.1f}%)")
    logger.info(f"  Medium (0.5-0.8): {confidence_stats['medium']} ({confidence_stats['medium']/total*100:.1f}%)")
    logger.info(f"  Low (<0.5): {confidence_stats['low']} ({confidence_stats['low']/total*100:.1f}%)")
    
    # Show sample of correct detections
    logger.info("\n✅ Sample Correct Detections:")
    correct_samples = [r for r in results if r['correct']][:5]
    for i, result in enumerate(correct_samples, 1):
        logger.info(f"  {i}. [{result['detected']}] {result['review'][:50]}... (conf: {result['confidence']:.2f})")
    
    # Show all incorrect detections
    incorrect = [r for r in results if not r['correct']]
    if incorrect:
        logger.info(f"\n❌ All Incorrect Detections ({len(incorrect)}):")
        for i, result in enumerate(incorrect, 1):
            logger.info(f"  {i}. Expected {result['expected']}, got {result['detected']}: {result['review'][:50]}...")
    
    return {
        'accuracy': accuracy,
        'total': total,
        'correct': correct,
        'by_language': by_language,
        'method_stats': dict(method_stats),
        'confidence_stats': confidence_stats
    }

def main():
    """Run the real-world language detection test"""
    
    # Fetch real reviews
    reviews_data = fetch_real_reviews(sample_size=50)
    
    if not reviews_data:
        logger.error("Failed to fetch reviews")
        return False
    
    logger.info(f"\n✓ Fetched {len(reviews_data)} reviews for testing")
    
    # Test language detection
    results = test_language_detection(reviews_data)
    
    # Final verdict
    logger.info("\n" + "="*80)
    if results['accuracy'] >= 80:
        logger.info(f"🎉 EXCELLENT: {results['accuracy']:.1f}% accuracy on real reviews!")
    elif results['accuracy'] >= 70:
        logger.info(f"✅ GOOD: {results['accuracy']:.1f}% accuracy on real reviews")
    elif results['accuracy'] >= 60:
        logger.info(f"⚠️ ACCEPTABLE: {results['accuracy']:.1f}% accuracy on real reviews")
    else:
        logger.info(f"❌ NEEDS IMPROVEMENT: {results['accuracy']:.1f}% accuracy on real reviews")
    
    return results['accuracy'] >= 70

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)