#!/usr/bin/env python3
"""
Universal Google Play Review Collection Script
Collects reviews from any app and stores in separate collections

Usage:
    python collect_reviews.py com.nianticlabs.pokemongo
    python collect_reviews.py "https://play.google.com/store/apps/details?id=com.nianticlabs.pokemongo&hl=en"
    python collect_reviews.py com.instagram.android --days 7 --locales en es pt
"""

import asyncio
import sys
import os
import argparse
import re
from urllib.parse import urlparse, parse_qs
from datetime import datetime

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from pavel.ingestion.google_play import GooglePlayIngester

def extract_app_id(input_string: str) -> str:
    """
    Extract app_id from either direct ID or Google Play URL
    
    Examples:
        com.nianticlabs.pokemongo -> com.nianticlabs.pokemongo
        https://play.google.com/store/apps/details?id=com.nianticlabs.pokemongo&hl=en -> com.nianticlabs.pokemongo
    """
    # Check if it's a URL
    if input_string.startswith('http://') or input_string.startswith('https://'):
        try:
            parsed = urlparse(input_string)
            params = parse_qs(parsed.query)
            if 'id' in params:
                return params['id'][0]
            else:
                raise ValueError(f"No 'id' parameter found in URL: {input_string}")
        except Exception as e:
            raise ValueError(f"Failed to parse URL: {e}")
    
    # Otherwise treat as direct app_id
    # Validate it looks like an app_id (contains dots, starts with letter or com/org/net)
    if '.' in input_string and re.match(r'^[a-zA-Z][a-zA-Z0-9._]*$', input_string):
        return input_string
    else:
        raise ValueError(f"Invalid app_id format: {input_string}")

async def main():
    """
    Collect reviews from Google Play for any app
    """
    parser = argparse.ArgumentParser(
        description='Collect Google Play reviews for any app',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using app ID directly
    python collect_reviews.py com.nianticlabs.pokemongo
    
    # Using Google Play URL
    python collect_reviews.py "https://play.google.com/store/apps/details?id=com.instagram.android"
    
    # With custom parameters
    python collect_reviews.py com.spotify.music --days 7 --locales en es pt
        """
    )
    parser.add_argument('app_input', 
                       help='App ID or Google Play URL')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to look back (default: 30)')
    parser.add_argument('--locales', nargs='+', 
                       help='Specific locales to collect (default: all major markets)')
    
    args = parser.parse_args()
    
    # Extract app_id from input
    try:
        app_id = extract_app_id(args.app_input)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    days_back = args.days
    
    # Use provided locales or default to all major markets
    if args.locales:
        locales = args.locales
    else:
        locales = [
            'en',  # English
            'ru',  # Russian  
            'es',  # Spanish
            'pt',  # Portuguese
            'fr',  # French
            'de',  # German
            'it',  # Italian
            'ja',  # Japanese
            'ko',  # Korean
            'zh',  # Chinese
            'ar',  # Arabic
            'hi',  # Hindi
            'id',  # Indonesian
            'tr',  # Turkish
            'pl',  # Polish
            'nl',  # Dutch
            'sv',  # Swedish
            'no',  # Norwegian
            'da',  # Danish
            'fi',  # Finnish
            'cs',  # Czech
            'hu',  # Hungarian
            'ro',  # Romanian
            'uk',  # Ukrainian
        ]
    
    # Use app_id directly as collection name (MongoDB supports dots in collection names)
    collection_name = app_id
    
    print(f"=" * 60)
    print(f"GOOGLE PLAY REVIEW COLLECTION")
    print(f"=" * 60)
    print(f"App ID: {app_id}")
    print(f"Collection: {collection_name}")
    print(f"Languages: {len(locales)} markets")
    print(f"Period: Last {days_back} days")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 60)
    
    ingester = None
    try:
        # Create ingester with custom collection name
        ingester = GooglePlayIngester(collection_name=collection_name)
        
        # Collect reviews
        stats = await ingester.ingest_batch_history(
            app_id=app_id,
            locales=locales,
            days_back=days_back
        )
        
        # Print results
        print(f"\n{'=' * 60}")
        print("COLLECTION RESULTS")
        print(f"{'=' * 60}")
        
        total_fetched = 0
        total_new = 0
        total_duplicates = 0
        
        for stat in stats:
            if stat.total_fetched > 0:
                print(f"\n{stat.locale.upper():5} | Fetched: {stat.total_fetched:5} | New: {stat.new_reviews:5} | Dupes: {stat.duplicates:5}")
                total_fetched += stat.total_fetched
                total_new += stat.new_reviews
                total_duplicates += stat.duplicates
        
        print(f"\n{'=' * 60}")
        print(f"TOTAL | Fetched: {total_fetched:5} | New: {total_new:5} | Dupes: {total_duplicates:5}")
        print(f"{'=' * 60}")
        
        # Get summary from database
        summary = ingester.get_ingestion_summary(app_id)
        print(f"\nDATABASE SUMMARY:")
        print(f"  Total reviews: {summary.get('total_reviews', 0)}")
        if summary.get('avg_score'):
            print(f"  Average score: {summary['avg_score']:.2f}")
        if summary.get('latest_review'):
            print(f"  Latest review: {summary['latest_review']}")
        if summary.get('earliest_review'):
            print(f"  Earliest review: {summary['earliest_review']}")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during collection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ingester:
            ingester.close()
            print("\n✅ Database connection closed")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main())