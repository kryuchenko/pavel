"""
Google Play review ingestion with rate limiting and error handling.

Supports both batch historical ingestion and incremental updates.
"""

import asyncio
from typing import List, Dict, Optional, AsyncIterator, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
import traceback

from google_play_scraper import reviews, Sort
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

from pavel.core.config import get_config
from pavel.core.logger import get_logger
from .rate_limiter import RateLimiter, RateLimit

logger = get_logger(__name__)

@dataclass 
class IngestionStats:
    """Statistics for ingestion run"""
    app_id: str
    locale: str
    total_fetched: int = 0
    new_reviews: int = 0
    duplicates: int = 0
    errors: int = 0
    start_time: datetime = None
    end_time: datetime = None
    
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
        
    def reviews_per_second(self) -> float:
        duration = self.duration_seconds()
        return self.total_fetched / duration if duration > 0 else 0.0

class GooglePlayIngester:
    """
    Google Play review ingester with comprehensive data preservation.
    
    Features:
    - Rate limiting per app
    - Batch and incremental modes  
    - Complete data preservation (all 11 fields)
    - Duplicate prevention via composite keys
    - Multi-locale support
    - Robust error handling
    """
    
    def __init__(self, mongo_client: Optional[MongoClient] = None):
        self.config = get_config()
        self.mongo_client = mongo_client or self._get_mongo_client()
        self.db = self.mongo_client[self.config.MONGODB_DATABASE]
        self.collection = self.db.reviews
        self.rate_limiter = RateLimiter()
        
        # Ensure indexes for performance
        self._ensure_indexes()
        
    def _get_mongo_client(self) -> MongoClient:
        """Get MongoDB client from config"""
        return MongoClient(self.config.MONGODB_URI)
        
    def _ensure_indexes(self):
        """Ensure required indexes exist"""
        indexes = [
            [("appId", 1), ("at", -1)],  # App + date queries
            [("appId", 1), ("score", 1)], # Rating analysis
            [("createdAt", -1)],          # Recent reviews
            [("flags.isRecent", 1)]       # Recent flag queries
        ]
        
        for index_spec in indexes:
            try:
                self.collection.create_index(index_spec, background=True)
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
                
    def _transform_review(self, review: Dict, app_id: str) -> Dict:
        """
        Transform raw Google Play review to PAVEL format.
        
        Preserves all original data + adds PAVEL metadata.
        """
        review_id = review.get('reviewId', '')
        now = datetime.now(timezone.utc)
        
        # Handle timezone-aware datetime comparison
        review_date = review.get('at')
        if review_date and not review_date.tzinfo:
            review_date = review_date.replace(tzinfo=timezone.utc)
            
        doc = {
            "_id": f"{app_id}:{review_id}",  # Composite key for global uniqueness
            "appId": app_id,
            "reviewId": review_id,
            "userName": review.get('userName', ''),
            "content": review.get('content', ''),
            "score": review.get('score', 0),
            "thumbsUpCount": review.get('thumbsUpCount', 0), 
            "reviewCreatedVersion": review.get('reviewCreatedVersion'),
            "at": review_date,
            "replyContent": review.get('replyContent'),
            "repliedAt": review.get('repliedAt'),
            "appVersion": review.get('appVersion'),
            
            # PAVEL metadata
            "createdAt": now,
            "updatedAt": now,
            "processingStatus": "ingested",
            "flags": {
                "hasReply": bool(review.get('replyContent')),
                "isRecent": (now - (review_date or now)).days < 30 if review_date else False,
                "hasVersion": bool(review.get('appVersion')),
                "isPositive": review.get('score', 0) >= 4,
                "isNegative": review.get('score', 0) <= 2
            },
            
            # Complete data preservation
            "rawData": {
                "original": review,
                "ingested_at": now.isoformat(),
                "source": "google-play-scraper"
            }
        }
        
        return doc
        
    async def _fetch_reviews_batch(
        self, 
        app_id: str, 
        locale: str = 'en',
        count: int = 200,
        sort: Sort = Sort.NEWEST,
        continuation_token: Optional[str] = None
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        Fetch a batch of reviews with rate limiting.
        
        Returns: (reviews, continuation_token)
        """
        await self.rate_limiter.wait_if_needed(app_id)
        
        try:
            logger.debug(f"Fetching {count} reviews for {app_id} ({locale})")
            
            result, next_token = reviews(
                app_id,
                lang=locale,
                country=locale[:2] if len(locale) > 2 else locale,
                sort=sort,
                count=count,
                continuation_token=continuation_token
            )
            
            logger.info(f"Fetched {len(result)} reviews for {app_id} ({locale})")
            return result, next_token
            
        except Exception as e:
            logger.error(f"Error fetching reviews for {app_id}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
            
    async def _insert_reviews(self, reviews: List[Dict], stats: IngestionStats) -> None:
        """Insert reviews into MongoDB with duplicate handling"""
        if not reviews:
            return
            
        try:
            # Transform all reviews
            docs = [self._transform_review(review, stats.app_id) for review in reviews]
            
            # Bulk insert with ordered=False for duplicate tolerance
            result = self.collection.insert_many(docs, ordered=False)
            stats.new_reviews += len(result.inserted_ids)
            
        except Exception as e:
            # Handle bulk write errors (including duplicates)
            if hasattr(e, 'details'):
                # Count successful inserts vs duplicates
                inserted_count = e.details.get('nInserted', 0)
                write_errors = e.details.get('writeErrors', [])
                
                stats.new_reviews += inserted_count
                
                duplicate_count = sum(1 for err in write_errors 
                                    if err.get('code') == 11000)  # Duplicate key error
                stats.duplicates += duplicate_count
                stats.errors += len(write_errors) - duplicate_count
                
                logger.info(f"Bulk insert: {inserted_count} new, {duplicate_count} duplicates, "
                          f"{len(write_errors) - duplicate_count} errors")
            else:
                stats.errors += len(docs)
                logger.error(f"Bulk insert error: {e}")
                
    async def ingest_batch_history(
        self, 
        app_id: str,
        locales: List[str] = ['en', 'ru'],
        days_back: int = 90,
        batch_size: int = 200
    ) -> List[IngestionStats]:
        """
        Ingest historical reviews (batch mode).
        
        Fetches up to 90 days of history across multiple locales.
        """
        stats_list = []
        
        for locale in locales:
            stats = IngestionStats(app_id=app_id, locale=locale)
            stats.start_time = datetime.now(timezone.utc)
            
            logger.info(f"Starting batch ingestion for {app_id} ({locale}) - {days_back} days")
            
            try:
                continuation_token = None
                total_batches = 0
                max_batches = 50  # Prevent infinite loops
                
                while total_batches < max_batches:
                    # Fetch batch
                    batch_reviews, continuation_token = await self._fetch_reviews_batch(
                        app_id=app_id,
                        locale=locale, 
                        count=batch_size,
                        sort=Sort.NEWEST,
                        continuation_token=continuation_token
                    )
                    
                    if not batch_reviews:
                        logger.info(f"No more reviews for {app_id} ({locale})")
                        break
                        
                    # Check if we've gone too far back
                    oldest_review = min(batch_reviews, key=lambda r: r.get('at', datetime.min))
                    oldest_date = oldest_review.get('at')
                    
                    if oldest_date:
                        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
                        if oldest_date.replace(tzinfo=timezone.utc) < cutoff_date:
                            logger.info(f"Reached {days_back} day cutoff for {app_id} ({locale})")
                            # Filter only recent reviews from this batch
                            batch_reviews = [r for r in batch_reviews 
                                           if r.get('at', datetime.min).replace(tzinfo=timezone.utc) >= cutoff_date]
                            
                    stats.total_fetched += len(batch_reviews)
                    
                    # Insert batch
                    await self._insert_reviews(batch_reviews, stats)
                    
                    total_batches += 1
                    logger.debug(f"Completed batch {total_batches} for {app_id} ({locale})")
                    
                    # Stop if we've reached the date cutoff or no continuation
                    if not continuation_token or (oldest_date and 
                        oldest_date.replace(tzinfo=timezone.utc) < cutoff_date):
                        break
                        
                logger.info(f"Batch ingestion completed for {app_id} ({locale}): "
                          f"{stats.total_fetched} fetched, {stats.new_reviews} new, "
                          f"{stats.duplicates} duplicates")
                          
            except Exception as e:
                logger.error(f"Batch ingestion error for {app_id} ({locale}): {e}")
                stats.errors += 1
                
            finally:
                stats.end_time = datetime.now(timezone.utc)
                stats_list.append(stats)
                
        return stats_list
        
    async def ingest_incremental(
        self,
        app_id: str,
        locales: List[str] = ['en', 'ru'],
        batch_size: int = 100
    ) -> List[IngestionStats]:
        """
        Incremental ingestion (fetch only new reviews since last run).
        
        Looks at the most recent review date in DB and fetches newer ones.
        """
        stats_list = []
        
        for locale in locales:
            stats = IngestionStats(app_id=app_id, locale=locale)
            stats.start_time = datetime.now(timezone.utc)
            
            logger.info(f"Starting incremental ingestion for {app_id} ({locale})")
            
            try:
                # Find the most recent review date for this app/locale
                latest_review = self.collection.find_one(
                    {"appId": app_id},
                    sort=[("at", -1)]
                )
                
                since_date = None
                if latest_review and latest_review.get('at'):
                    since_date = latest_review['at']
                    logger.info(f"Last review date: {since_date}")
                else:
                    logger.info("No previous reviews found, fetching recent")
                    
                # Fetch new reviews (always start from most recent)
                batch_reviews, _ = await self._fetch_reviews_batch(
                    app_id=app_id,
                    locale=locale,
                    count=batch_size,
                    sort=Sort.NEWEST
                )
                
                if not batch_reviews:
                    logger.info(f"No new reviews for {app_id} ({locale})")
                else:
                    # Filter only truly new reviews
                    if since_date:
                        new_reviews = [
                            r for r in batch_reviews
                            if r.get('at') and r['at'].replace(tzinfo=timezone.utc) > since_date.replace(tzinfo=timezone.utc)
                        ]
                    else:
                        new_reviews = batch_reviews
                        
                    stats.total_fetched = len(batch_reviews)
                    
                    if new_reviews:
                        await self._insert_reviews(new_reviews, stats)
                        logger.info(f"Incremental: {len(new_reviews)} new reviews for {app_id} ({locale})")
                    else:
                        logger.info(f"No truly new reviews for {app_id} ({locale})")
                        
            except Exception as e:
                logger.error(f"Incremental ingestion error for {app_id} ({locale}): {e}")
                stats.errors += 1
                
            finally:
                stats.end_time = datetime.now(timezone.utc)
                stats_list.append(stats)
                
        return stats_list
        
    def get_ingestion_summary(self, app_id: str) -> Dict:
        """Get ingestion summary for app"""
        pipeline = [
            {"$match": {"appId": app_id}},
            {"$group": {
                "_id": None,
                "total_reviews": {"$sum": 1},
                "latest_review": {"$max": "$at"},
                "earliest_review": {"$min": "$at"},
                "avg_score": {"$avg": "$score"},
                "has_replies": {"$sum": {"$cond": ["$flags.hasReply", 1, 0]}}
            }}
        ]
        
        result = list(self.collection.aggregate(pipeline))
        if result:
            summary = result[0]
            summary["app_id"] = app_id
            summary.pop("_id")
            return summary
        else:
            return {"app_id": app_id, "total_reviews": 0}
            
    def close(self):
        """Close MongoDB connection"""
        if self.mongo_client:
            self.mongo_client.close()