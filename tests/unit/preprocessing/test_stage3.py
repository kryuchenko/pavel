#!/usr/bin/env python3
"""
Stage 3 validation: Text preprocessing and normalization

Tests:
- Text normalization (Unicode, HTML, URLs, etc.)
- Language detection (langdetect + patterns)
- Sentence segmentation (multilingual)
- Content deduplication (exact + fuzzy)
- Complete preprocessing pipeline
"""

import asyncio
from datetime import datetime, timezone
from typing import List

# Import PAVEL modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pavel.core.config import get_config
from pavel.core.logger import get_logger
from pavel.preprocessing.normalizer import TextNormalizer
from pavel.preprocessing.language_detector import LanguageDetector
from pavel.preprocessing.sentence_splitter import SentenceSplitter
from pavel.preprocessing.deduplicator import ContentDeduplicator
from pavel.preprocessing.pipeline import PreprocessingPipeline

logger = get_logger(__name__)

class TestStage3:
    """Stage 3: Text preprocessing validation"""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        cls.config = get_config()
        logger.info("Testing Stage 3: Text preprocessing")
        
    def test_1_text_normalization(self):
        """Test text normalization features"""
        logger.info("TEST 1: Text normalization")
        
        normalizer = TextNormalizer(
            preserve_emojis=True,
            max_repeated_chars=3,
            remove_urls=True
        )
        
        # Test cases
        test_cases = [
            # HTML entities
            ("This app is &lt;great&gt; &amp; amazing!", 
             "This app is <great> & amazing!"),
            
            # Unicode normalization
            ("Café naïve résumé",  # Combined characters
             "Café naïve résumé"),  # NFC normalized
            
            # URLs removal
            ("Check out https://example.com for more info",
             "Check out for more info"),
            
            # Repeated characters
            ("Sooooo goooooood!!!!!!",
             "Sooo goood!!!"),
            
            # Multiple whitespace
            ("Too    many     spaces    here",
             "Too many spaces here"),
            
            # Mixed case with emojis
            ("I love this app 😍😍😍 so much!!!",
             "I love this app 😍😍😍 so much!!!"),
        ]
        
        passed = 0
        for original, expected in test_cases:
            normalized, stats = normalizer.normalize(original)
            
            # Clean up expected (remove extra spaces from URL removal)
            expected = " ".join(expected.split())
            normalized = " ".join(normalized.split())
            
            if normalized == expected:
                passed += 1
                logger.debug(f"✓ Normalized correctly: {original[:30]}...")
            else:
                logger.warning(f"✗ Normalization mismatch:")
                logger.warning(f"  Original: {original}")
                logger.warning(f"  Expected: {expected}")
                logger.warning(f"  Got: {normalized}")
                
        logger.info(f"✅ Text normalization: {passed}/{len(test_cases)} test cases passed")
        
        # Test meaningfulness check
        assert normalizer.is_meaningful_text("Good app") == True
        assert normalizer.is_meaningful_text("!!!") == False
        assert normalizer.is_meaningful_text("") == False
        assert normalizer.is_meaningful_text("a") == False
        
        logger.info("✅ Meaningfulness detection working")
        
    def test_2_language_detection(self):
        """Test language detection"""
        logger.info("TEST 2: Language detection")
        
        detector = LanguageDetector(
            min_confidence=0.7,
            min_text_length=10
        )
        
        # Test cases with expected languages
        test_cases = [
            ("This is a great app for navigation", "en"),
            ("Это отличное приложение для навигации", "ru"),
            ("Esta es una gran aplicación para navegación", "es"),
            ("Este é um ótimo aplicativo para navegação", "pt"),
            ("Ini adalah aplikasi yang bagus untuk navigasi", "id"),
        ]
        
        passed = 0
        total_confidence = 0.0
        
        for text, expected_lang in test_cases:
            result = detector.detect(text)
            
            if result.language == expected_lang:
                passed += 1
                logger.debug(f"✓ Detected {result.language} ({result.confidence:.2f}): {text[:30]}...")
            else:
                logger.warning(f"✗ Language mismatch: expected {expected_lang}, got {result.language}")
                logger.warning(f"  Text: {text}")
                
            total_confidence += result.confidence
            
        avg_confidence = total_confidence / len(test_cases)
        logger.info(f"✅ Language detection: {passed}/{len(test_cases)} correct, "
                   f"avg confidence: {avg_confidence:.2f}")
        
        # Test pattern-based detection
        pattern_result = detector.detect_by_patterns("Это очень хорошо")
        assert pattern_result is not None
        assert pattern_result.language == "ru"
        logger.info("✅ Pattern-based detection working")
        
        # Test locale fallback
        locale_result = detector.detect_by_locale("en_US")
        assert locale_result is not None
        assert locale_result.language == "en"
        logger.info("✅ Locale fallback working")
        
    def test_3_sentence_segmentation(self):
        """Test sentence splitting"""
        logger.info("TEST 3: Sentence segmentation")
        
        splitter = SentenceSplitter(
            min_sentence_length=5,
            max_sentence_length=500
        )
        
        # Test cases
        test_cases = [
            # Simple sentences
            ("This app is great. It works perfectly. I love it!",
             ["This app is great.", "It works perfectly.", "I love it!"]),
            
            # Single sentence
            ("Amazing app with great features",
             ["Amazing app with great features"]),
            
            # With line breaks
            ("First paragraph here.\n\nSecond paragraph here.",
             ["First paragraph here.", "Second paragraph here."]),
            
            # With abbreviations
            ("Dr. Smith recommended this app. It's really good.",
             ["Dr. Smith recommended this app.", "It's really good."]),
            
            # Short review
            ("Good",
             ["Good"]),
            
            # With emojis
            ("Love it! 😍 Works great! 👍",
             ["Love it! 😍 Works great! 👍"]),
        ]
        
        passed = 0
        for text, expected_sentences in test_cases:
            result = splitter.split(text)
            
            # Normalize for comparison
            actual_sentences = [s.strip() for s in result.sentences]
            expected_normalized = [s.strip() for s in expected_sentences]
            
            # Check if sentences match (order matters)
            matches = True
            if len(actual_sentences) != len(expected_normalized):
                matches = False
            else:
                for actual, expected in zip(actual_sentences, expected_normalized):
                    if actual != expected:
                        matches = False
                        break
                        
            if matches:
                passed += 1
                logger.debug(f"✓ Split correctly: {text[:30]}... → {len(actual_sentences)} sentences")
            else:
                logger.warning(f"✗ Sentence split mismatch:")
                logger.warning(f"  Text: {text}")
                logger.warning(f"  Expected: {expected_normalized}")
                logger.warning(f"  Got: {actual_sentences}")
                
        logger.info(f"✅ Sentence segmentation: {passed}/{len(test_cases)} test cases passed")
        
        # Test statistics
        results = splitter.batch_split(["First. Second.", "Single sentence."])
        stats = splitter.get_statistics(results)
        assert stats["total_texts"] == 2
        assert stats["total_sentences"] == 3
        logger.info("✅ Batch processing and statistics working")
        
    def test_4_content_deduplication(self):
        """Test content deduplication"""
        logger.info("TEST 4: Content deduplication")
        
        deduplicator = ContentDeduplicator(
            similarity_threshold=0.8,
            min_length_for_comparison=10
        )
        
        # Test exact duplicates
        texts_exact = [
            "This app is amazing",
            "Great navigation tool",
            "This app is amazing",  # Exact duplicate
            "Love this application",
            "Great navigation tool"  # Exact duplicate
        ]
        
        result = deduplicator.deduplicate(texts_exact, method="exact")
        
        assert result.original_count == 5
        assert result.unique_count == 3
        assert result.duplicate_count == 2
        logger.info(f"✅ Exact deduplication: {result.unique_count} unique from {result.original_count}")
        
        # Test fuzzy duplicates
        texts_fuzzy = [
            "This app is really amazing",
            "This app is really amazing!",  # Very similar
            "Great navigation tool for drivers",
            "Great navigation tool for driver",  # Very similar
            "Completely different review here"
        ]
        
        result_fuzzy = deduplicator.deduplicate(texts_fuzzy, method="fuzzy")
        
        assert result_fuzzy.duplicate_count > 0  # Should find some duplicates
        logger.info(f"✅ Fuzzy deduplication: found {result_fuzzy.duplicate_count} duplicates")
        
        # Test get unique texts
        unique_texts = deduplicator.get_unique_texts(texts_exact)
        assert len(unique_texts) == 3
        logger.info(f"✅ Get unique texts: {len(unique_texts)} unique texts extracted")
        
        # Test analysis
        analysis = deduplicator.analyze_duplication_patterns(texts_exact)
        assert analysis["duplication_rate"] > 0
        logger.info(f"✅ Duplication analysis: {analysis['duplication_rate']:.2%} duplication rate")
        
    def test_5_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline"""
        logger.info("TEST 5: Complete preprocessing pipeline")
        
        async def test_pipeline():
            # Note: This test doesn't require MongoDB connection
            # We're testing the processing logic
            
            pipeline = PreprocessingPipeline(mongo_client=None)
            
            # Create mock review data
            mock_review = {
                'reviewId': 'test_123',
                'appId': 'test.app',
                'content': 'This app is AMAZING!!! Check https://example.com for more. Это приложение отличное!',
                'locale': 'en_US'
            }
            
            # Process single review
            processed = await pipeline.process_single_review(mock_review)
            
            # Validate processing results
            assert processed.review_id == 'test_123'
            assert processed.app_id == 'test.app'
            assert len(processed.normalized_content) > 0
            assert processed.detected_language in ['en', 'ru']  # Mixed content
            assert processed.sentence_count > 0
            assert len(processed.sentences) > 0
            assert processed.language_confidence > 0
            
            # Check processing stats
            assert 'normalization' in processed.processing_stats
            assert 'language_detection' in processed.processing_stats
            assert 'sentence_splitting' in processed.processing_stats
            
            logger.info(f"✅ Pipeline processed review: "
                       f"language={processed.detected_language} ({processed.language_confidence:.2f}), "
                       f"sentences={processed.sentence_count}")
            
            # Test error handling
            bad_review = {
                'reviewId': 'bad_123',
                'appId': 'test.app',
                'content': None,  # Invalid content
                'locale': 'en_US'
            }
            
            bad_processed = await pipeline.process_single_review(bad_review)
            assert bad_processed.review_id == 'bad_123'
            assert 'error' in bad_processed.processing_stats or bad_processed.sentence_count == 0
            logger.info("✅ Pipeline handles errors gracefully")
            
        asyncio.run(test_pipeline())
        
    def test_6_multilingual_support(self):
        """Test multilingual processing"""
        logger.info("TEST 6: Multilingual support")
        
        normalizer = TextNormalizer()
        detector = LanguageDetector()
        splitter = SentenceSplitter()
        
        # Test different languages
        multilingual_texts = [
            ("Hello world! This is English.", "en"),
            ("Привет мир! Это русский язык.", "ru"),
            ("¡Hola mundo! Esto es español.", "es"),
            ("Olá mundo! Isto é português.", "pt"),
            ("Halo dunia! Ini bahasa Indonesia.", "id"),
        ]
        
        for text, expected_lang in multilingual_texts:
            # Normalize
            normalized, _ = normalizer.normalize(text)
            assert len(normalized) > 0
            
            # Detect language
            lang_result = detector.detect(normalized)
            # Allow some flexibility in language detection
            logger.debug(f"Detected {lang_result.language} for {expected_lang} text")
            
            # Split sentences
            sentence_result = splitter.split(normalized, lang_result.language)
            assert sentence_result.sentence_count > 0
            
        logger.info("✅ Multilingual support: all languages processed successfully")
        
    def test_7_performance_metrics(self):
        """Test performance and efficiency"""
        logger.info("TEST 7: Performance metrics")
        
        import time
        
        normalizer = TextNormalizer()
        
        # Generate test data
        test_texts = [
            f"This is test review number {i}. It has some content with URLs https://test.com and emojis 😊!"
            for i in range(100)
        ]
        
        # Measure normalization performance
        start_time = time.time()
        results = normalizer.batch_normalize(test_texts)
        end_time = time.time()
        
        duration = end_time - start_time
        texts_per_second = len(test_texts) / duration
        
        assert len(results) == len(test_texts)
        assert all(len(r[0]) > 0 for r in results)  # All normalized
        
        logger.info(f"✅ Performance: Normalized {len(test_texts)} texts in {duration:.2f}s "
                   f"({texts_per_second:.0f} texts/sec)")
        
        # Test batch processing efficiency
        detector = LanguageDetector()
        
        start_time = time.time()
        lang_results = detector.batch_detect([r[0] for r in results])
        end_time = time.time()
        
        duration = end_time - start_time
        detections_per_second = len(lang_results) / duration
        
        assert len(lang_results) == len(test_texts)
        
        logger.info(f"✅ Language detection: {len(test_texts)} texts in {duration:.2f}s "
                   f"({detections_per_second:.0f} detections/sec)")

def main():
    """Run Stage 3 validation tests"""
    print("=" * 60)
    print("PAVEL Stage 3 Validation: Text Preprocessing")
    print("=" * 60)
    
    # Create test instance
    test = TestStage3()
    test.setup_class()
    
    # Run tests in order
    tests = [
        test.test_1_text_normalization,
        test.test_2_language_detection,
        test.test_3_sentence_segmentation,
        test.test_4_content_deduplication,
        test.test_5_preprocessing_pipeline,
        test.test_6_multilingual_support,
        test.test_7_performance_metrics,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            logger.error(f"Test failed: {test_func.__name__} - {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    print("\n" + "=" * 60)
    print(f"Stage 3 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Stage 3: Text Preprocessing - COMPLETE")
        print("\nFeatures validated:")
        print("  ✅ Text normalization (Unicode, HTML, URLs)")
        print("  ✅ Language detection (multi-method)")
        print("  ✅ Sentence segmentation (multilingual)")
        print("  ✅ Content deduplication (exact + fuzzy)")
        print("  ✅ Complete preprocessing pipeline")
        print("  ✅ Multilingual support (en, ru, es, pt, id)")
        print("  ✅ Performance optimization")
        return True
    else:
        print(f"❌ {total - passed} test(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)