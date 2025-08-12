#!/usr/bin/env python3
"""
Real-world test with actual Google Play data over 2 weeks.

This test:
1. Fetches real reviews from Google Play for the last 2 weeks
2. Generates real embeddings using Stage 4 (E5 multilingual)
3. Runs smart anomaly detection with cluster dynamics
4. Shows actual operational vs product issues
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Import all PAVEL stages
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pavel.ingestion.google_play import GooglePlayIngester
from pavel.ingestion.rate_limiter import RateLimiter, RateLimit
from pavel.preprocessing.pipeline import PreprocessingPipeline
from pavel.embeddings.embedding_pipeline import EmbeddingPipeline, PipelineConfig
from pavel.clustering.smart_detection_pipeline import SmartDetectionPipeline
from pavel.clustering.dynamic_cluster_detector import IssueCategory

from pavel.core.logger import get_logger
from pavel.core.app_config import get_default_app_id, get_default_app_info

logger = get_logger(__name__)


async def fetch_real_reviews(app_id: str, days: int = 14) -> List[Dict[str, Any]]:
    """
    Fetch real reviews from Google Play for the specified period.
    """
    print(f"\n📱 Fetching real reviews for {app_id} from last {days} days...")
    
    # Use google_play_scraper directly for simplicity
    from google_play_scraper import reviews as gp_reviews, Sort
    import time
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch reviews (will get latest available)
        all_reviews = []
        
        # Fetch reviews for multiple languages
        for lang in ['en', 'ru', 'es']:  # Multi-language (reduced for speed)
            print(f"   Fetching {lang} reviews...")
            
            try:
                # Direct call to google_play_scraper
                result, _ = gp_reviews(
                    app_id,
                    lang=lang,
                    country='us',
                    sort=Sort.NEWEST,
                    count=100
                )
                reviews = result
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(1)
                
                if reviews:
                    # Filter by date if timestamps available
                    filtered = []
                    for review in reviews:
                        # Parse review date if available
                        review_date = review.get('at')
                        if review_date:
                            if isinstance(review_date, str):
                                try:
                                    review_date = datetime.fromisoformat(review_date.replace('Z', '+00:00'))
                                except:
                                    review_date = datetime.now()  # Fallback
                            
                            if review_date >= start_date:
                                filtered.append(review)
                        else:
                            # If no date, include it (assume recent)
                            filtered.append(review)
                    
                    all_reviews.extend(filtered)
                    print(f"      Got {len(filtered)} recent reviews")
                
            except Exception as e:
                print(f"      Error fetching {lang} reviews: {e}")
                continue
        
        print(f"   Total reviews collected: {len(all_reviews)}")
        
        # Convert to our format
        formatted_reviews = []
        for i, review in enumerate(all_reviews):
            formatted_reviews.append({
                'review_id': review.get('reviewId', f'review_{i}'),
                'app_id': app_id,
                'content': review.get('content', ''),
                'rating': review.get('score', 3),
                'created_at': review.get('at', datetime.now()),
                'locale': review.get('lang', 'en'),
                'userName': review.get('userName', 'Anonymous'),
                'thumbsUpCount': review.get('thumbsUpCount', 0),
                'replyContent': review.get('replyContent'),
                'repliedAt': review.get('repliedAt')
            })
        
        return formatted_reviews
        
    except Exception as e:
        logger.error(f"Failed to fetch reviews: {e}")
        print(f"❌ Error fetching reviews: {e}")
        return []


async def run_real_world_test(app_id: Optional[str] = None):
    """
    Run complete PAVEL pipeline on real Google Play data.
    
    Default app configured in .env: PAVEL_DEFAULT_APP_ID
    """
    # Get app info from environment
    if app_id is None:
        app_info = get_default_app_info()
        app_id = app_info['app_id']
    else:
        app_info = {'name': app_id, 'url': f"https://play.google.com/store/apps/details?id={app_id}"}
    
    print("🚀 PAVEL REAL-WORLD TEST WITH ACTUAL GOOGLE PLAY DATA")
    print("=" * 70)
    print(f"📱 Target App: {app_info['name']} ({app_id})")
    print(f"🔗 Play Store: {app_info['url']}")
    print(f"📅 Period: Last 14 days")
    print(f"🧠 Using: Real embeddings + Smart clustering")
    print("=" * 70)
    
    # Stage 1-2: Fetch real reviews
    reviews = await fetch_real_reviews(app_id, days=14)
    
    if not reviews:
        print("❌ No reviews fetched. Please check app_id and connection.")
        return False
    
    print(f"\n✅ Stage 1-2: Collected {len(reviews)} real reviews")
    
    # Show sample reviews
    print("\n📝 Sample reviews:")
    for review in reviews[:3]:
        print(f"   ⭐ {review['rating']} - {review['content'][:100]}...")
    
    # Stage 3: Preprocessing (simplified - just language detection)
    print("\n🔧 Stage 3: Preprocessing...")
    
    # Simple language detection
    languages = set()
    for review in reviews:
        # Detect language based on locale or content
        lang = review.get('locale', 'en')[:2]  # Get first 2 chars (en_US -> en)
        review['language'] = lang
        languages.add(lang)
    
    print(f"   Languages detected: {set(r['language'] for r in reviews)}")
    
    # Stage 4: Generate real embeddings
    print("\n🧠 Stage 4: Generating real embeddings...")
    embedding_config = PipelineConfig(
        embedding_model="intfloat/multilingual-e5-small",
        batch_size=32,
        enable_preprocessing=False  # Already preprocessed
    )
    embedding_pipeline = EmbeddingPipeline(embedding_config)
    
    # Generate embeddings using the pipeline
    try:
        result = await embedding_pipeline.process_app_reviews(
            app_id=app_id,
            reviews=reviews
        )
        
        # Extract embeddings from results
        embeddings_map = {}
        for batch in result.embedding_batches:
            for embedding_result in batch.embeddings:
                embeddings_map[embedding_result.review_id] = embedding_result.embedding
        
        print(f"   Generated {len(embeddings_map)} embeddings")
        
    except Exception as e:
        # Fallback: generate embeddings individually
        print(f"   Pipeline failed ({e}), using fallback...")
        embeddings_map = {}
        
        for i, review in enumerate(reviews):
            try:
                result = await embedding_pipeline.process_single_review(
                    review_text=review['content'],
                    app_id=app_id,
                    review_id=review['review_id']
                )
                embeddings_map[review['review_id']] = result.embedding
                
                if (i + 1) % 50 == 0:
                    print(f"   Generated embeddings: {i+1}/{len(reviews)}")
                    
            except Exception as e2:
                print(f"   Error on review {i}: {e2}")
                continue
    
    print(f"✅ Generated {len(embeddings_map)} embeddings")
    
    # Stage 5: Smart anomaly detection
    print("\n🔍 Stage 5: Smart Anomaly Detection...")
    smart_pipeline = SmartDetectionPipeline(
        embedding_pipeline=embedding_pipeline,
        history_weeks=2,
        min_reviews_for_analysis=20  # Lower threshold for real data
    )
    
    try:
        # Run analysis
        result = await smart_pipeline.analyze_reviews(
            app_id=app_id,
            reviews=reviews,
            end_date=datetime.now()
        )
        
        print("\n" + "=" * 70)
        print("📊 REAL DATA ANALYSIS RESULTS")
        print("=" * 70)
        
        # Overall health
        print(f"\n🏥 APP HEALTH SCORE: {result.overall_health_score:.1f}/100")
        
        # Current week stats
        current = result.current_week
        print(f"\n📅 CURRENT WEEK ANALYSIS:")
        print(f"   Reviews analyzed: {current.total_reviews}")
        print(f"   Clusters found: {current.clusters_found}")
        print(f"   Anomalies detected: {len(current.anomalies)}")
        
        # Show cluster categories
        print(f"\n🏷️ CLUSTER CATEGORIES FOUND:")
        category_counts = {}
        for cat in current.cluster_categories.values():
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        for category, count in category_counts.items():
            print(f"   {category.value}: {count} clusters")
        
        # Operational vs Product breakdown
        print(f"\n🔧 ISSUE BREAKDOWN:")
        if result.operational_alerts:
            print(f"   🚨 OPERATIONAL ISSUES: {len(result.operational_alerts)}")
            for alert in result.operational_alerts[:3]:
                keywords = ', '.join(alert.cluster_profile.dominant_keywords[:3])
                print(f"      • Cluster {alert.cluster_id}: {keywords}")
                print(f"        {alert.explanation}")
                print(f"        → {alert.suggested_action}")
        else:
            print("   ✅ No operational issues detected")
        
        if result.product_bugs:
            print(f"\n   🐛 PRODUCT ISSUES: {len(result.product_bugs)}")
            for bug in result.product_bugs[:3]:
                keywords = ', '.join(bug.cluster_profile.dominant_keywords[:3])
                print(f"      • Cluster {bug.cluster_id}: {keywords}")
                print(f"        {bug.explanation}")
                print(f"        → {bug.suggested_action}")
        else:
            print("   ✅ No product bugs detected")
        
        # Week-over-week changes
        if len(result.previous_weeks) > 0:
            print(f"\n📈 WEEK-OVER-WEEK CHANGES:")
            for metric, value in current.week_over_week_changes.items():
                if isinstance(value, (int, float)):
                    print(f"   {metric}: {value:+.1f}{'%' if 'pct' in metric else ''}")
        
        # Critical changes
        if result.critical_changes:
            print(f"\n⚠️ CRITICAL CHANGES:")
            for change in result.critical_changes[:5]:
                keywords = ', '.join(change.cluster_profile.dominant_keywords[:3])
                print(f"   • {change.anomaly_type} in '{keywords}' cluster")
                print(f"     Severity: {change.severity:.1f}")
                print(f"     Change: {change.change_magnitude:.1f}%")
        
        # Emerging topics (new issues)
        if current.emerging_topics:
            print(f"\n🆕 EMERGING TOPICS:")
            for topic in current.emerging_topics[:5]:
                print(f"   • {topic}")
        
        # Cluster trends
        if result.cluster_trends:
            print(f"\n📊 CLUSTER TRENDS:")
            for cluster_id, trend in list(result.cluster_trends.items())[:5]:
                print(f"   Cluster {cluster_id}:")
                print(f"      Size: {trend['size_trend']} ({trend['size_change_pct']:+.1f}%)")
                print(f"      Rating: {trend['rating_trend']} ({trend['rating_change']:+.1f})")
        
        # Immediate actions
        print(f"\n💡 RECOMMENDED ACTIONS:")
        for action in result.immediate_actions[:5]:
            print(f"   {action}")
        
        # Performance metrics
        print(f"\n⚡ PERFORMANCE:")
        print(f"   Processing time: {result.processing_time_ms:.0f}ms")
        print(f"   Reviews/second: {len(reviews) / (result.processing_time_ms/1000):.1f}")
        print(f"   Embeddings cached: {result.embeddings_cached}")
        
        # Save results for analysis
        results_file = f"real_test_results_{app_id.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        save_data = {
            'app_id': app_id,
            'total_reviews': len(reviews),
            'health_score': result.overall_health_score,
            'operational_issues': len(result.operational_alerts),
            'product_bugs': len(result.product_bugs),
            'critical_changes': len(result.critical_changes),
            'clusters_found': current.clusters_found,
            'processing_time_ms': result.processing_time_ms,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n📄 Results saved to: {results_file}")
        
        print("\n" + "=" * 70)
        print("✅ REAL-WORLD TEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """
    Main function to run real-world test.
    """
    print("🔬 PAVEL REAL-WORLD TEST WITH GOOGLE PLAY DATA")
    print("=" * 70)
    
    # Use default app from .env configuration
    app_info = get_default_app_info()
    print(f"\n🎯 Testing with: {app_info['name']}")
    print(f"🔗 {app_info['url']}")
    
    success = await run_real_world_test()  # Uses default from .env
    
    if success:
        print(f"✅ {app_info['name']} analysis complete!")
    else:
        print(f"❌ {app_info['name']} analysis failed")
    
    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())