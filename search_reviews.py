#!/usr/bin/env python3
"""
CLI utility for semantic search in reviews.
Demonstrates vector search capabilities.
"""

import asyncio
import argparse
import sys
from typing import Optional

# Add project to path
sys.path.insert(0, 'src')

from pavel.search.vector_search import VectorSearchEngine, SearchQuery

async def main():
    """Main CLI interface for vector search."""
    parser = argparse.ArgumentParser(
        description='Semantic search in Google Play reviews',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic search
    python search_reviews.py "bugs and crashes"
    
    # Search with filters
    python search_reviews.py "good game" --limit 5 --min-similarity 0.8
    
    # Language-specific search
    python search_reviews.py "плохая игра" --locale ru
    
    # Find similar reviews
    python search_reviews.py --similar-to some_review_id
    
    # Search by sentiment
    python search_reviews.py --sentiment positive --limit 10
    
    # Find issues
    python search_reviews.py --issues --locale es
    
    # Show collection stats
    python search_reviews.py --stats
        """
    )
    
    # Main search options
    parser.add_argument('query', nargs='?', help='Search query text')
    parser.add_argument('--app-id', default='com.nianticlabs.pokemongo',
                       help='App ID to search (default: Pokemon GO)')
    parser.add_argument('--limit', type=int, default=10,
                       help='Maximum number of results (default: 10)')
    parser.add_argument('--min-similarity', type=float, default=0.0,
                       help='Minimum similarity threshold (0.0-1.0)')
    parser.add_argument('--locale', help='Filter by language (en, ru, es, etc.)')
    parser.add_argument('--score', type=int, help='Filter by review score (1-5)')
    
    # Special search modes
    parser.add_argument('--similar-to', help='Find reviews similar to this review ID')
    parser.add_argument('--sentiment', choices=['positive', 'negative'],
                       help='Search by sentiment')
    parser.add_argument('--issues', action='store_true',
                       help='Search for technical issues')
    parser.add_argument('--stats', action='store_true',
                       help='Show collection statistics')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.query, args.similar_to, args.sentiment, args.issues, args.stats]):
        parser.error("Must provide query text or use --similar-to, --sentiment, --issues, or --stats")
    
    # Initialize search engine
    print(f"🔍 Initializing semantic search for {args.app_id}...")
    search_engine = VectorSearchEngine(collection_name=args.app_id)
    
    try:
        if args.stats:
            await show_stats(search_engine)
        elif args.similar_to:
            await find_similar(search_engine, args.similar_to, args.limit, args.verbose)
        elif args.sentiment:
            await search_sentiment(search_engine, args.sentiment, args.limit, args.locale, args.verbose)
        elif args.issues:
            await search_issues(search_engine, args.limit, args.locale, args.verbose)
        else:
            await search_text(search_engine, args.query, args.limit, args.min_similarity, 
                            args.locale, args.score, args.verbose, args.json)
    
    finally:
        search_engine.close()

async def search_text(search_engine: VectorSearchEngine, 
                     query_text: str,
                     limit: int,
                     min_similarity: float,
                     locale: Optional[str],
                     score: Optional[int],
                     verbose: bool,
                     json_output: bool):
    """Perform text-based semantic search."""
    print(f"🔎 Searching: '{query_text}'")
    print("-" * 50)
    
    # Build filter
    filter_params = {}
    if locale:
        filter_params["locale"] = locale
    if score:
        filter_params["score"] = score
    
    # Create query
    query = SearchQuery(
        text=query_text,
        limit=limit,
        min_similarity=min_similarity,
        filter_params=filter_params if filter_params else None
    )
    
    # Execute search
    results = await search_engine.search(query)
    
    if not results:
        print("❌ No results found")
        return
    
    # Display results
    if json_output:
        import json
        json_results = []
        for r in results:
            json_results.append({
                "similarity": r.similarity,
                "score": r.score,
                "locale": r.locale,
                "content": r.content,
                "review_id": r.review_id
            })
        print(json.dumps(json_results, indent=2, ensure_ascii=False))
    else:
        print(f"✅ Found {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"{i:2d}. [{result.locale.upper()}] {result.similarity:.3f} - {result.score}⭐")
            content = result.content[:100] if len(result.content) > 100 else result.content
            print(f"    \"{content}{'...' if len(result.content) > 100 else ''}\"")
            
            if verbose:
                print(f"    Review ID: {result.review_id}")
                print(f"    App ID: {result.app_id}")
            
            print()

async def find_similar(search_engine: VectorSearchEngine,
                      review_id: str,
                      limit: int,
                      verbose: bool):
    """Find reviews similar to a specific review."""
    print(f"🔍 Finding reviews similar to: {review_id}")
    print("-" * 50)
    
    try:
        results = await search_engine.find_similar_reviews(review_id, limit)
        
        if not results:
            print("❌ No similar reviews found")
            return
        
        print(f"✅ Found {len(results)} similar reviews:\n")
        for i, result in enumerate(results, 1):
            print(f"{i:2d}. [{result.locale.upper()}] {result.similarity:.3f} - {result.score}⭐")
            content = result.content[:100] if len(result.content) > 100 else result.content
            print(f"    \"{content}{'...' if len(result.content) > 100 else ''}\"")
            
            if verbose:
                print(f"    Review ID: {result.review_id}")
            
            print()
            
    except ValueError as e:
        print(f"❌ Error: {e}")

async def search_sentiment(search_engine: VectorSearchEngine,
                          sentiment: str,
                          limit: int,
                          locale: Optional[str],
                          verbose: bool):
    """Search reviews by sentiment."""
    print(f"😊 Searching {sentiment} reviews" + (f" in {locale}" if locale else ""))
    print("-" * 50)
    
    results = await search_engine.search_by_sentiment(sentiment, limit, locale)
    
    if not results:
        print("❌ No results found")
        return
    
    print(f"✅ Found {len(results)} {sentiment} reviews:\n")
    for i, result in enumerate(results, 1):
        emoji = "😊" if sentiment == "positive" else "😞"
        print(f"{i:2d}. [{result.locale.upper()}] {result.similarity:.3f} - {result.score}⭐ {emoji}")
        content = result.content[:100] if len(result.content) > 100 else result.content
        print(f"    \"{content}{'...' if len(result.content) > 100 else ''}\"")
        
        if verbose:
            print(f"    Review ID: {result.review_id}")
        
        print()

async def search_issues(search_engine: VectorSearchEngine,
                       limit: int,
                       locale: Optional[str],
                       verbose: bool):
    """Search for technical issues."""
    print(f"🐛 Searching for technical issues" + (f" in {locale}" if locale else ""))
    print("-" * 50)
    
    results = await search_engine.search_issues(limit, locale)
    
    if not results:
        print("❌ No issues found")
        return
    
    print(f"✅ Found {len(results)} reviews mentioning issues:\n")
    for i, result in enumerate(results, 1):
        print(f"{i:2d}. [{result.locale.upper()}] {result.similarity:.3f} - {result.score}⭐ 🐛")
        content = result.content[:100] if len(result.content) > 100 else result.content
        print(f"    \"{content}{'...' if len(result.content) > 100 else ''}\"")
        
        if verbose:
            print(f"    Review ID: {result.review_id}")
        
        print()

async def show_stats(search_engine: VectorSearchEngine):
    """Show collection statistics."""
    print("📊 Collection Statistics")
    print("=" * 50)
    
    stats = search_engine.get_collection_stats()
    
    print(f"Collection: {stats['collection_name']}")
    print(f"Total reviews: {stats['total_reviews']:,}")
    print(f"Reviews with embeddings: {stats['reviews_with_embeddings']:,}")
    print(f"Embedding coverage: {stats['embedding_coverage']:.1%}")
    
    print(f"\n📈 Language Distribution:")
    for lang_stat in stats['language_distribution']:
        lang = lang_stat['_id']
        count = lang_stat['count']
        percentage = count / stats['total_reviews'] * 100
        print(f"  {lang}: {count:,} reviews ({percentage:.1f}%)")
    
    print(f"\n⭐ Score Distribution:")
    for score_stat in stats['score_distribution']:
        score = score_stat['_id']
        count = score_stat['count']
        percentage = count / stats['total_reviews'] * 100
        stars = "⭐" * score
        print(f"  {score} {stars}: {count:,} reviews ({percentage:.1f}%)")

if __name__ == "__main__":
    asyncio.run(main())