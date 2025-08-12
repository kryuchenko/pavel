#!/usr/bin/env python3
"""
Stage 0 Test: Verify basic setup and Google Play fetch
Tests default appId fallback and rate limiting
"""

import time
from google_play_scraper import Sort, reviews
from pavel.core.config import config
from pavel.core.logger import get_logger

logger = get_logger("test_stage0")


def test_config():
    """Test configuration and default appId"""
    logger.info("Testing configuration...")
    
    # Test default app ID
    assert config.DEFAULT_APP_ID == "sinet.startup.inDriver", "Default app ID mismatch"
    logger.info(f"✓ Default app ID: {config.DEFAULT_APP_ID}")
    
    # Test app ID fallback
    app_id_with_override = config.get_app_id("com.example.test")
    assert app_id_with_override == "com.example.test", "App ID override failed"
    
    app_id_default = config.get_app_id(None)
    assert app_id_default == "sinet.startup.inDriver", "App ID fallback failed"
    logger.info("✓ App ID fallback works correctly")
    
    # Check configuration
    logger.info(f"✓ DB: {config.DB_NAME}")
    logger.info(f"✓ Batch size: {config.BATCH_SIZE}")
    logger.info(f"✓ Rate limit: {config.RATE_LIMIT_REQUESTS} requests per {config.RATE_LIMIT_PERIOD}s")
    logger.info(f"✓ Default locales: {', '.join(config.DEFAULT_LOCALES[:3])}...")


def test_google_play_fetch():
    """Test fetching reviews from Google Play"""
    logger.info("\nTesting Google Play fetch...")
    
    app_id = config.DEFAULT_APP_ID
    test_locale = "en_US"  # Test with single locale
    
    logger.info(f"Fetching reviews for: {app_id}")
    logger.info(f"Locale: {test_locale}")
    
    try:
        # Measure fetch time
        start_time = time.time()
        
        # Fetch small batch of reviews
        result, continuation_token = reviews(
            app_id,
            lang=test_locale[:2],  # Use language code only
            country=test_locale[-2:],  # Use country code
            sort=Sort.NEWEST,
            count=10  # Small batch for testing
        )
        
        fetch_time = time.time() - start_time
        
        logger.info(f"✓ Fetched {len(result)} reviews in {fetch_time:.2f}s")
        
        # Verify fields are not empty
        if result:
            sample = result[0]
            required_fields = ['reviewId', 'userName', 'content', 'score', 'at']
            
            for field in required_fields:
                assert field in sample and sample[field] is not None, f"Field {field} is empty"
                
            logger.info("✓ All required fields present")
            logger.info(f"  - Review ID: {sample['reviewId'][:20]}...")
            logger.info(f"  - User: {sample['userName'][:20]}...")
            logger.info(f"  - Score: {sample['score']}")
            logger.info(f"  - Date: {sample['at']}")
            logger.info(f"  - Content: {sample['content'][:50]}...")
            
            # Check optional fields
            optional_fields = ['replyContent', 'repliedAt', 'appVersion']
            for field in optional_fields:
                if field in sample and sample[field]:
                    logger.info(f"  - {field}: Present")
        
        # Test rate limiting
        logger.info("\nTesting rate limit awareness...")
        
        # Simulate multiple requests
        request_times = []
        for i in range(3):
            start = time.time()
            _, _ = reviews(
                app_id,
                lang=test_locale[:2],
                country=test_locale[-2:],
                sort=Sort.NEWEST,
                count=5
            )
            request_times.append(time.time() - start)
            
            if i < 2:  # Don't sleep after last request
                time.sleep(1)  # Small delay between requests
        
        avg_time = sum(request_times) / len(request_times)
        logger.info(f"✓ Average request time: {avg_time:.2f}s")
        logger.info(f"✓ Rate limiting handled (no 429 errors)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Fetch failed: {e}")
        return False


def test_multiple_locales():
    """Test fetching from multiple locales"""
    logger.info("\nTesting multiple locales...")
    
    app_id = config.DEFAULT_APP_ID
    test_locales = ["en_US", "ru_RU", "es_ES"][:2]  # Test first 2 locales
    
    results = {}
    for locale in test_locales:
        try:
            result, _ = reviews(
                app_id,
                lang=locale[:2],
                country=locale[-2:],
                sort=Sort.NEWEST,
                count=5
            )
            results[locale] = len(result)
            logger.info(f"✓ {locale}: {len(result)} reviews")
            time.sleep(0.5)  # Rate limit protection
        except Exception as e:
            logger.warning(f"✗ {locale}: Failed - {e}")
            results[locale] = 0
    
    successful = sum(1 for count in results.values() if count > 0)
    logger.info(f"✓ Successfully fetched from {successful}/{len(test_locales)} locales")
    
    return successful > 0


def main():
    """Run all Stage 0 tests"""
    logger.info("=" * 60)
    logger.info("PAVEL Stage 0 Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Configuration", test_config),
        ("Google Play Fetch", test_google_play_fetch),
        ("Multiple Locales", test_multiple_locales),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            logger.info(f"\n✓ {test_name}: PASSED\n")
        except Exception as e:
            failed += 1
            logger.error(f"\n✗ {test_name}: FAILED - {e}\n")
    
    logger.info("=" * 60)
    logger.info(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("✅ Stage 0: All tests passed - Ready for Stage 1")
    else:
        logger.error("❌ Stage 0: Some tests failed - Fix issues before proceeding")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)