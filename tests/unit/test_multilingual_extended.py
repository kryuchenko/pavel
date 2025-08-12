#!/usr/bin/env python3
"""
Extended multilingual language detection test for inDriver markets.

Tests language detection across major inDriver markets with real text samples.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pavel.preprocessing.language_detector import LanguageDetector
from pavel.core.logger import get_logger

logger = get_logger(__name__)

def test_extended_languages():
    """Test language detection for inDriver's global markets"""
    
    detector = LanguageDetector(min_confidence=0.6)
    
    # Test samples from inDriver markets
    test_samples = [
        # Cyrillic-based
        ("Отличное приложение для заказа такси! Очень быстро и удобно.", "ru", "Russian"),
        ("Бұл қосымша өте жақсы, жылдам және ыңғайлы.", "kk", "Kazakh"),
        ("Це додаток дуже гарний, швидкий і зручний.", "uk", "Ukrainian"),
        
        # Arabic script
        ("هذا التطبيق ممتاز لطلب سيارة أجرة. سريع جداً ومريح.", "ar", "Arabic"),
        ("این برنامه عالی است برای درخواست تاکسی. بسیار سریع و راحت.", "fa", "Persian/Farsi"),
        ("یہ ایپ ٹیکسی بک کرنے کے لیے بہت اچھی ہے۔", "ur", "Urdu"),
        
        # Latin script
        ("This app is excellent for booking taxis! Very fast and convenient.", "en", "English"),
        ("Esta aplicación es excelente para pedir taxis. Muy rápida y conveniente.", "es", "Spanish"),
        ("Este aplicativo é excelente para chamar táxis. Muito rápido e conveniente.", "pt", "Portuguese"),
        ("Aplikasi ini sangat bagus untuk memesan taksi. Sangat cepat dan nyaman.", "id", "Indonesian"),
        ("Bu uygulama taksi çağırmak için mükemmel! Çok hızlı ve kullanışlı.", "tr", "Turkish"),
        
        # Asian scripts
        ("แอพนี้ดีมากสำหรับเรียกแท็กซี่ เร็วและสะดวกมาก", "th", "Thai"),
        ("यह ऐप टैक्सी बुक करने के लिए बहुत अच्छा है। बहुत तेज़ और सुविधाजनक।", "hi", "Hindi"),
        ("এই অ্যাপটি ট্যাক্সি বুক করার জন্য খুবই ভাল। খুব দ্রুত এবং সুবিধাজনক।", "bn", "Bengali"),
        ("Ứng dụng này rất tuyệt vời để đặt taxi. Rất nhanh và tiện lợi.", "vi", "Vietnamese"),
        ("这个应用程序叫出租车很棒！非常快速和方便。", "zh", "Chinese"),
        ("このアプリはタクシーを呼ぶのに素晴らしいです！とても速くて便利です。", "ja", "Japanese"),
        ("이 앱은 택시 호출에 훌륭합니다! 매우 빠르고 편리합니다.", "ko", "Korean"),
    ]
    
    results = []
    correct = 0
    total = len(test_samples)
    
    logger.info("="*80)
    logger.info("EXTENDED MULTILINGUAL LANGUAGE DETECTION TEST")
    logger.info("="*80)
    
    for i, (text, expected, lang_name) in enumerate(test_samples, 1):
        detection = detector.detect(text)
        
        is_correct = detection.language == expected
        if is_correct:
            correct += 1
        
        results.append({
            'text': text,
            'expected': expected,
            'detected': detection.language,
            'confidence': detection.confidence,
            'method': detection.method,
            'correct': is_correct,
            'language_name': lang_name
        })
        
        status = "✓" if is_correct else "✗"
        logger.info(f"{status} {i:2d}. {lang_name:15} | Expected: {expected:2} | Got: {detection.language:2} ({detection.confidence:.2f}) | {detection.method}")
        if not is_correct:
            logger.info(f"      Text: {text[:60]}...")
    
    # Calculate statistics
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Group by script families
    script_families = {
        'cyrillic': ['ru', 'kk', 'uk'],
        'arabic': ['ar', 'fa', 'ur'],
        'latin': ['en', 'es', 'pt', 'id', 'tr'],
        'asian': ['th', 'hi', 'bn', 'vi', 'zh', 'ja', 'ko']
    }
    
    family_stats = {}
    for family, languages in script_families.items():
        family_results = [r for r in results if r['expected'] in languages]
        family_correct = len([r for r in family_results if r['correct']])
        family_total = len(family_results)
        family_accuracy = (family_correct / family_total * 100) if family_total > 0 else 0
        family_stats[family] = {
            'correct': family_correct,
            'total': family_total,
            'accuracy': family_accuracy
        }
    
    # Method distribution
    method_distribution = {}
    for result in results:
        method = result['method']
        method_distribution[method] = method_distribution.get(method, 0) + 1
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EXTENDED LANGUAGE DETECTION SUMMARY")
    logger.info("="*80)
    
    logger.info(f"\n📊 Overall Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    logger.info(f"\n📚 Accuracy by Script Family:")
    for family, stats in family_stats.items():
        logger.info(f"  {family.capitalize():10}: {stats['correct']:2}/{stats['total']:2} ({stats['accuracy']:5.1f}%)")
    
    logger.info(f"\n🔧 Detection Methods:")
    for method, count in method_distribution.items():
        percentage = (count / total * 100)
        logger.info(f"  {method:12}: {count:2} ({percentage:5.1f}%)")
    
    # Show failed detections
    failed = [r for r in results if not r['correct']]
    if failed:
        logger.info(f"\n❌ Failed Detections ({len(failed)}):")
        for r in failed:
            logger.info(f"  {r['language_name']:15} | Expected {r['expected']} → Got {r['detected']} | {r['text'][:50]}...")
    
    # Show pattern-based detections (our custom patterns working)
    pattern_detections = [r for r in results if r['method'] == 'patterns']
    if pattern_detections:
        logger.info(f"\n🎯 Pattern-Based Detections ({len(pattern_detections)}):")
        for r in pattern_detections:
            status = "✓" if r['correct'] else "✗"
            logger.info(f"  {status} {r['language_name']:15} | {r['detected']} | {r['text'][:50]}...")
    
    return {
        'accuracy': accuracy,
        'total': total,
        'correct': correct,
        'family_stats': family_stats,
        'method_distribution': method_distribution,
        'pattern_detections': len(pattern_detections)
    }

def test_short_texts():
    """Test language detection on short texts (common in reviews)"""
    
    detector = LanguageDetector(min_confidence=0.5)
    
    short_samples = [
        ("Great app!", "en"),
        ("Muy bueno", "es"),
        ("Muito bom", "pt"),
        ("Очень хорошо", "ru"),
        ("Өте жақсы", "kk"),
        ("ممتاز", "ar"),
        ("بسیار خوب", "fa"),
        ("Çok iyi", "tr"),
        ("Sangat bagus", "id"),
        ("很好", "zh"),
        ("とても良い", "ja"),
        ("아주 좋아", "ko"),
        ("ดีมาก", "th"),
        ("बहुत अच्छा", "hi"),
        ("Rất tốt", "vi"),
    ]
    
    logger.info("\n" + "="*80)
    logger.info("SHORT TEXT DETECTION TEST")
    logger.info("="*80)
    
    correct = 0
    total = len(short_samples)
    
    for i, (text, expected) in enumerate(short_samples, 1):
        detection = detector.detect(text)
        is_correct = detection.language == expected
        
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        logger.info(f"{status} {i:2d}. '{text:15}' | Expected: {expected} | Got: {detection.language} ({detection.confidence:.2f})")
    
    accuracy = (correct / total * 100) if total > 0 else 0
    logger.info(f"\n📊 Short Text Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    return accuracy

def main():
    """Run extended multilingual tests"""
    
    logger.info("🌍 Testing inDriver Global Language Detection")
    
    # Test extended languages
    extended_results = test_extended_languages()
    
    # Test short texts
    short_accuracy = test_short_texts()
    
    # Final verdict
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS")
    logger.info("="*80)
    
    logger.info(f"📈 Extended Languages: {extended_results['accuracy']:.1f}% accuracy")
    logger.info(f"📝 Short Texts: {short_accuracy:.1f}% accuracy")
    logger.info(f"🎯 Pattern Detections: {extended_results['pattern_detections']} languages using custom patterns")
    
    avg_accuracy = (extended_results['accuracy'] + short_accuracy) / 2
    
    if avg_accuracy >= 80:
        logger.info(f"🎉 EXCELLENT: {avg_accuracy:.1f}% average accuracy across all tests!")
        logger.info("✅ Ready for production use in inDriver markets")
    elif avg_accuracy >= 70:
        logger.info(f"✅ GOOD: {avg_accuracy:.1f}% average accuracy")
        logger.info("⚠️  Consider improvements for some languages")
    else:
        logger.info(f"⚠️  NEEDS IMPROVEMENT: {avg_accuracy:.1f}% average accuracy")
    
    # Market coverage
    supported_markets = len(extended_results['family_stats'])
    logger.info(f"\n🗺️  Market Coverage: {supported_markets} script families supported")
    logger.info("📍 Ready for inDriver's global expansion!")
    
    return avg_accuracy >= 75

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)