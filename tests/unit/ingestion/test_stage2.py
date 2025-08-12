#!/usr/bin/env python3
"""
Stage 2 validation: Data ingestion from Google Play Store

Tests:
- Batch historical ingestion (90 days)
- Incremental ingestion (new reviews only)
- Rate limiting functionality
- Multi-locale support
- Error handling and recovery
- Job scheduling and coordination
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from typing import List

# Import PAVEL modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pavel.core.config import get_config
from pavel.core.logger import get_logger
from pavel.ingestion.google_play import GooglePlayIngester, IngestionStats
from pavel.ingestion.rate_limiter import RateLimiter, RateLimit
from pavel.ingestion.batch_processor import BatchProcessor, BatchJob
from pavel.ingestion.scheduler import IncrementalScheduler, ScheduleConfig

logger = get_logger(__name__)

class TestStage2:
    """Stage 2: Data ingestion validation"""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        cls.config = get_config()
        # Use a well-known app with many reviews for testing
        cls.app_id = "com.whatsapp"  # WhatsApp has many reviews
        logger.info(f"Testing Stage 2 with app_id: {cls.app_id}")
        
    def test_1_rate_limiter(self):
        """Test rate limiting functionality"""
        logger.info("TEST 1: Rate limiter functionality")
        
        # Create rate limiter with strict limits for testing
        rate_limit = RateLimit(requests_per_minute=5, burst_limit=2, cooldown_seconds=10)
        limiter = RateLimiter()
        limiter.set_app_limit(self.app_id, rate_limit)
        
        # Test initial state
        stats = limiter.get_stats(self.app_id)
        assert stats["requests_last_minute"] == 0
        assert stats["in_cooldown"] == False
        assert stats["utilization"] == 0.0
        
        # Simulate requests within burst limit
        async def test_burst():
            for i in range(2):  # Within burst limit
                await limiter.wait_if_needed(self.app_id)
                
            stats = limiter.get_stats(self.app_id)
            assert stats["requests_last_minute"] == 2
            assert stats["in_cooldown"] == False
            
            # Next request should trigger cooldown
            await limiter.wait_if_needed(self.app_id)
            stats = limiter.get_stats(self.app_id)
            assert stats["in_cooldown"] == True
            
        asyncio.run(test_burst())
        logger.info("✅ Rate limiter: burst limit and cooldown working")
        
    def test_2_google_play_ingester_basic(self):
        """Test basic Google Play ingester functionality"""
        logger.info("TEST 2: Google Play ingester basic functionality")
        
        async def test_ingester():
            ingester = GooglePlayIngester()
            
            try:
                # Test single batch fetch
                reviews, continuation = await ingester._fetch_reviews_batch(
                    app_id=self.app_id,
                    locale='en',
                    count=10
                )
                
                assert len(reviews) > 0, "Should fetch some reviews"
                assert all('reviewId' in r for r in reviews), "All reviews should have reviewId"
                assert all('content' in r for r in reviews), "All reviews should have content"
                
                # Test data transformation
                transformed = ingester._transform_review(reviews[0], self.app_id)
                
                # Validate PAVEL format
                assert transformed["_id"] == f"{self.app_id}:{reviews[0]['reviewId']}"
                assert transformed["appId"] == self.app_id
                assert "rawData" in transformed
                assert "flags" in transformed
                assert "createdAt" in transformed
                
                logger.info(f"✅ Fetched {len(reviews)} reviews, transformation working")
                
            finally:
                ingester.close()
                
        asyncio.run(test_ingester())
        
    def test_3_incremental_ingestion(self):
        """Test incremental ingestion (new reviews only)"""
        logger.info("TEST 3: Incremental ingestion")
        
        async def test_incremental():
            ingester = GooglePlayIngester()
            
            try:
                # Run incremental ingestion
                stats_list = await ingester.ingest_incremental(
                    app_id=self.app_id,
                    locales=['en'],
                    batch_size=50
                )
                
                assert len(stats_list) == 1, "Should have stats for 1 locale"
                stats = stats_list[0]
                
                assert stats.app_id == self.app_id
                assert stats.locale == 'en'
                assert stats.total_fetched >= 0
                assert stats.duration_seconds() > 0
                
                # Get summary
                summary = ingester.get_ingestion_summary(self.app_id)
                assert summary["app_id"] == self.app_id
                assert summary["total_reviews"] >= 0
                
                logger.info(f"✅ Incremental: {stats.total_fetched} fetched, "
                          f"{stats.new_reviews} new, {stats.duplicates} duplicates")
                
            finally:
                ingester.close()
                
        asyncio.run(test_incremental())
        
    def test_4_batch_historical_ingestion(self):
        """Test batch historical ingestion (limited for testing)"""
        logger.info("TEST 4: Batch historical ingestion")
        
        async def test_batch():
            ingester = GooglePlayIngester()
            
            try:
                # Run batch ingestion with short history for testing
                stats_list = await ingester.ingest_batch_history(
                    app_id=self.app_id,
                    locales=['en'],
                    days_back=7,  # Just 1 week for testing
                    batch_size=100
                )
                
                assert len(stats_list) == 1, "Should have stats for 1 locale"
                stats = stats_list[0]
                
                assert stats.app_id == self.app_id
                assert stats.locale == 'en'
                assert stats.total_fetched >= 0
                assert stats.duration_seconds() > 0
                
                if stats.total_fetched > 0:
                    assert stats.reviews_per_second() > 0
                    
                logger.info(f"✅ Batch: {stats.total_fetched} fetched, "
                          f"{stats.new_reviews} new, {stats.reviews_per_second():.1f} reviews/sec")
                
            finally:
                ingester.close()
                
        asyncio.run(test_batch())
        
    def test_5_batch_processor(self):
        """Test batch processing coordinator"""
        logger.info("TEST 5: Batch processor coordination")
        
        async def test_processor():
            processor = BatchProcessor(max_concurrent=2)
            
            try:
                # Create test jobs
                jobs = [
                    BatchJob(app_id=self.app_id, locales=['en'], mode='incremental', batch_size=20),
                ]
                
                # Add a second test app if available
                test_app_2 = "com.google.android.apps.maps"  # Google Maps
                jobs.append(BatchJob(app_id=test_app_2, locales=['en'], mode='incremental', batch_size=20))
                
                # Process jobs
                results = await processor.process_jobs(jobs)
                
                assert len(results) == len(jobs), "Should have result for each job"
                
                successful_results = [r for r in results if r.success]
                assert len(successful_results) >= 1, "At least one job should succeed"
                
                # Generate report
                report = processor.generate_report(results)
                
                assert "summary" in report
                assert "apps" in report
                assert report["summary"]["total_jobs"] == len(jobs)
                
                logger.info(f"✅ Batch processor: {len(successful_results)}/{len(results)} jobs successful")
                logger.info(f"   Total reviews: {report['summary']['total_new_reviews']}")
                
            finally:
                await processor.close()
                
        asyncio.run(test_processor())
        
    def test_6_scheduler_configuration(self):
        """Test incremental scheduler configuration"""
        logger.info("TEST 6: Scheduler configuration")
        
        scheduler = IncrementalScheduler()
        
        try:
            # Add app configurations
            config1 = ScheduleConfig(
                app_id=self.app_id,
                locales=['en'],
                schedule_type='hourly',
                enabled=True
            )
            scheduler.add_app(config1)
            
            config2 = ScheduleConfig(
                app_id="com.google.android.apps.maps",
                locales=['en', 'ru'],
                schedule_type='daily',
                enabled=False
            )
            scheduler.add_app(config2)
            
            # Check status
            status = scheduler.get_status()
            
            assert status["total_apps"] == 2
            assert status["enabled_apps"] == 1
            assert self.app_id in status["apps"]
            assert status["apps"][self.app_id]["enabled"] == True
            assert status["apps"][self.app_id]["schedule_type"] == "hourly"
            
            # Test enable/disable
            scheduler.enable_app(self.app_id, False)
            status = scheduler.get_status()
            assert status["enabled_apps"] == 0
            
            logger.info("✅ Scheduler: configuration and status working")
            
        finally:
            asyncio.run(scheduler.close())
            
    def test_7_error_handling(self):
        """Test error handling with invalid app ID"""
        logger.info("TEST 7: Error handling")
        
        async def test_errors():
            ingester = GooglePlayIngester()
            
            try:
                # Test with invalid app ID
                invalid_app = "invalid.app.id.that.does.not.exist"
                
                stats_list = await ingester.ingest_incremental(
                    app_id=invalid_app,
                    locales=['en'],
                    batch_size=10
                )
                
                # Should handle error gracefully
                assert len(stats_list) == 1
                stats = stats_list[0]
                assert stats.app_id == invalid_app
                
                # Either no reviews or error count > 0
                assert stats.total_fetched == 0 or stats.errors > 0
                
                logger.info("✅ Error handling: graceful handling of invalid app ID")
                
            finally:
                ingester.close()
                
        asyncio.run(test_errors())
        
    def test_8_data_preservation(self):
        """Test complete data preservation"""
        logger.info("TEST 8: Data preservation validation")
        
        async def test_preservation():
            ingester = GooglePlayIngester()
            
            try:
                # Fetch a small batch
                reviews, _ = await ingester._fetch_reviews_batch(
                    app_id=self.app_id,
                    locale='en',
                    count=5
                )
                
                if not reviews:
                    logger.warning("No reviews fetched, skipping preservation test")
                    return
                    
                # Transform and check preservation
                review = reviews[0]
                transformed = ingester._transform_review(review, self.app_id)
                
                # Check all 11 Google Play fields are preserved
                google_play_fields = [
                    'reviewId', 'userName', 'content', 'score', 'thumbsUpCount',
                    'reviewCreatedVersion', 'at', 'replyContent', 'repliedAt', 'appVersion'
                ]
                
                for field in google_play_fields:
                    if field in review:  # Some fields might be None/missing
                        assert field in transformed, f"Field {field} not preserved"
                        
                # Check PAVEL extensions
                assert "rawData" in transformed
                assert "flags" in transformed
                assert "processingStatus" in transformed
                assert transformed["rawData"]["original"] == review
                
                # Check flags
                flags = transformed["flags"]
                assert "hasReply" in flags
                assert "isRecent" in flags
                assert "hasVersion" in flags
                assert "isPositive" in flags
                assert "isNegative" in flags
                
                logger.info("✅ Data preservation: all fields preserved with PAVEL extensions")
                
            finally:
                ingester.close()
                
        asyncio.run(test_preservation())

def main():
    """Run Stage 2 validation tests"""
    print("=" * 60)
    print("PAVEL Stage 2 Validation: Data Ingestion")
    print("=" * 60)
    
    # Create test instance
    test = TestStage2()
    test.setup_class()
    
    # Run tests in order
    tests = [
        test.test_1_rate_limiter,
        test.test_2_google_play_ingester_basic,
        test.test_3_incremental_ingestion,
        test.test_4_batch_historical_ingestion,
        test.test_5_batch_processor,
        test.test_6_scheduler_configuration,
        test.test_7_error_handling,
        test.test_8_data_preservation,
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
    print(f"Stage 2 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Stage 2: Data Ingestion - COMPLETE")
        print("\nFeatures validated:")
        print("  ✅ Rate limiting with burst control")
        print("  ✅ Google Play API integration")
        print("  ✅ Incremental ingestion (new reviews)")
        print("  ✅ Batch historical ingestion")
        print("  ✅ Concurrent job processing")
        print("  ✅ Scheduler configuration")
        print("  ✅ Error handling and recovery")
        print("  ✅ Complete data preservation")
        return True
    else:
        print(f"❌ {total - passed} test(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)