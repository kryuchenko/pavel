#!/usr/bin/env python3
"""
CLI tool for ingesting and embedding Google Play reviews.
Usage: python ingest_app.py com.example.app --months 3
"""

import asyncio
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import logging

# Add project to path
sys.path.insert(0, 'src')

from pavel.ingestion.google_play import GooglePlayIngester
from pavel.embeddings.embedding_generator import EmbeddingGenerator
from pavel.core.config import get_config
from pavel.core.logger import get_logger
from pymongo import MongoClient
from tqdm import tqdm
import time

logger = get_logger(__name__)

class AppIngester:
    """Ingest and embed reviews for a Google Play app."""
    
    def __init__(self):
        self.config = get_config()
        self.mongo_client = MongoClient(self.config.MONGODB_URI)
        self.db = self.mongo_client[self.config.MONGODB_DATABASE]
        self.ingester = GooglePlayIngester()
        self.embedding_generator = EmbeddingGenerator()
        
    async def ingest_and_embed(self, 
                               app_id: str, 
                               months: int = 3,
                               languages: Optional[list] = None) -> dict:
        """
        Ingest reviews and generate embeddings.
        
        Args:
            app_id: Google Play app ID
            months: Number of months to fetch
            languages: List of language codes (default: all major languages)
            
        Returns:
            Statistics dictionary
        """
        if not languages:
            languages = ['en', 'es', 'ru', 'pt', 'fr', 'de', 'it', 'id', 'tr', 'ar', 'ja', 'ko', 'zh']
        
        print(f"\n📱 Ingesting reviews for: {app_id}")
        print(f"📅 Period: last {months} months")
        print(f"🌍 Languages: {', '.join(languages)}")
        print("-" * 50)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)
        
        # Phase 1: Collect reviews with proper pagination and locale mapping
        print("\n📥 Phase 1: Collecting reviews...")
        
        from google_play_scraper import reviews as fetch_reviews, Sort
        
        # Correct locale to country mapping
        LOCALE_TO_COUNTRY = {
            "en": "us", "ru": "ru", "es": "es", "pt": "br", "fr": "fr", "de": "de",
            "it": "it", "id": "id", "tr": "tr", "ar": "sa", "ja": "jp", "ko": "kr",
            "zh": "tw"  # для Китая Google Play нет, обычно берут TW/HK
        }
        
        MAX_PER_LANG = 50000       # защитный лимит
        PAGE_SIZE = 200            # максимум для библиотеки
        SLEEP_MS = 100             # небольшая задержка против троттлинга
        
        total_reviews = 0
        collection = self.db[app_id]
        per_lang_stats = {}
        
        for lang in tqdm(languages, desc="Languages"):
            country = LOCALE_TO_COUNTRY.get(lang, "us")
            fetched_lang = 0
            token = None
            stop = False
            
            while not stop:
                try:
                    batch, token = fetch_reviews(
                        app_id,
                        lang=lang,
                        country=country,
                        sort=Sort.NEWEST,  # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: сортировка по новизне
                        count=PAGE_SIZE,
                        continuation_token=token
                    )
                    # Manual sleep instead of parameter
                    if SLEEP_MS > 0:
                        time.sleep(SLEEP_MS / 1000.0)
                except Exception as e:
                    logger.warning(f"[{lang}-{country}] fetch failed: {e}")
                    break
                
                if not batch:
                    break
                
                # Трансформация и фильтр по дате с ранним выходом
                transformed = []
                oldest_in_batch = None
                
                for r in batch:
                    tr = await self.ingester._transform_review(r, app_id, lang)
                    at = tr.get("at")
                    
                    if isinstance(at, str):
                        try:
                            at = datetime.fromisoformat(at.replace("Z", "+00:00"))
                        except Exception:
                            continue
                    if at and at.tzinfo:
                        at = at.replace(tzinfo=None)
                    
                    if at:
                        if oldest_in_batch is None or at < oldest_in_batch:
                            oldest_in_batch = at
                        if at >= start_date:
                            transformed.append(tr)
                
                # Апсертим только подходящее по дате
                for review in transformed:
                    review["_id"] = f"{app_id}:{review['reviewId']}"
                    review["appId"] = app_id
                    review["locale"] = lang
                    collection.update_one({"_id": review["_id"]}, {"$set": review}, upsert=True)
                
                total_reviews += len(transformed)
                fetched_lang += len(transformed)
                
                # если уже дошли до отзывов старше нужной даты — прекращаем пагинацию
                if oldest_in_batch and oldest_in_batch < start_date:
                    stop = True
                
                # защитный лимит
                if fetched_lang >= MAX_PER_LANG:
                    stop = True
                
                if not token:
                    break
            
            per_lang_stats[lang] = fetched_lang
            logger.info(f"[{lang}-{country}] collected {fetched_lang}")
        
        print(f"✅ Collected {total_reviews} reviews")
        print("Per-language:", per_lang_stats)
        
        # Phase 2: Generate embeddings
        print("\n🧠 Phase 2: Generating embeddings...")
        
        # Get reviews without embeddings
        reviews_to_embed = list(collection.find(
            {'embedding': {'$exists': False}},
            {'_id': 1, 'content': 1, 'locale': 1}
        ))
        
        if not reviews_to_embed:
            print("✅ All reviews already have embeddings")
            return {
                'app_id': app_id,
                'total_reviews': total_reviews,
                'embedded_reviews': 0,
                'languages': languages,
                'months': months
            }
        
        print(f"📊 Reviews to embed: {len(reviews_to_embed)}")
        
        # Process in batches
        batch_size = 32
        embedded_count = 0
        
        for i in tqdm(range(0, len(reviews_to_embed), batch_size), desc="Embedding batches"):
            batch = reviews_to_embed[i:i+batch_size]
            
            try:
                # Generate embeddings for batch
                texts = [r['content'] for r in batch]
                results = await self.embedding_generator.generate_batch_async(texts)
                
                # Update documents with embeddings
                for review, result in zip(batch, results):
                    if result.embedding is not None:
                        collection.update_one(
                            {'_id': review['_id']},
                            {'$set': {
                                'embedding': {
                                    'vector': result.embedding.tolist(),
                                    'model': result.model,
                                    'created_at': datetime.now()
                                }
                            }}
                        )
                        embedded_count += 1
                
            except Exception as e:
                logger.error(f"Failed to embed batch: {e}")
                continue
        
        print(f"✅ Generated {embedded_count} embeddings")
        
        # Create indexes
        print("\n🔍 Creating indexes...")
        collection.create_index([("at", -1)])
        collection.create_index([("score", 1)])
        collection.create_index([("locale", 1)])
        collection.create_index([("embedding.created_at", -1)])
        print("✅ Indexes created")
        
        # Final statistics
        stats = {
            'app_id': app_id,
            'total_reviews': total_reviews,
            'embedded_reviews': embedded_count,
            'languages': languages,
            'months': months,
            'collection_size': collection.count_documents({}),
            'embeddings_count': collection.count_documents({'embedding': {'$exists': True}})
        }
        
        print("\n📊 Final Statistics:")
        print(f"  • Total reviews collected: {stats['total_reviews']}")
        print(f"  • New embeddings created: {stats['embedded_reviews']}")
        print(f"  • Total collection size: {stats['collection_size']}")
        print(f"  • Total with embeddings: {stats['embeddings_count']}")
        print(f"  • Coverage: {stats['embeddings_count']/stats['collection_size']*100:.1f}%")
        
        return stats
    
    def close(self):
        """Close connections."""
        self.mongo_client.close()

async def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description='Ingest and embed Google Play reviews',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Ingest last 3 months of reviews
    python ingest_app.py com.example.app --months 3
    
    # Ingest 6 months with specific languages
    python ingest_app.py com.example.app --months 6 --languages en es ru
    
    # Ingest with custom MongoDB URI
    python ingest_app.py com.example.app --months 1 --db-uri mongodb://localhost:27017
        """
    )
    
    parser.add_argument('app_id', help='Google Play application ID')
    parser.add_argument('--months', type=int, default=3,
                       help='Number of months to fetch (default: 3)')
    parser.add_argument('--languages', nargs='+',
                       help='Language codes (default: all major languages)')
    parser.add_argument('--db-uri', help='MongoDB URI (overrides config)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    # Override DB URI if provided
    if args.db_uri:
        import os
        os.environ['PAVEL_DB_URI'] = args.db_uri
    
    # Run ingestion
    ingester = AppIngester()
    
    try:
        start_time = time.time()
        
        stats = await ingester.ingest_and_embed(
            app_id=args.app_id,
            months=args.months,
            languages=args.languages
        )
        
        elapsed = time.time() - start_time
        print(f"\n⏱️ Total time: {elapsed:.1f} seconds")
        print(f"✅ Successfully ingested and embedded reviews for {args.app_id}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    finally:
        ingester.close()

if __name__ == "__main__":
    asyncio.run(main())