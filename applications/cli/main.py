#\!/usr/bin/env python3
"""
PAVEL CLI Application

Command-line interface for PAVEL anomaly detection system.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pavel import get_logger, get_default_app_id

logger = get_logger(__name__)


async def ingest_reviews(app_id: str, locales: list, days: int = 90):
    """Ingest reviews from Google Play."""
    from pavel.ingestion.google_play import GooglePlayIngester
    
    ingester = GooglePlayIngester()
    
    print(f"🔍 Ingesting reviews for {app_id}")
    print(f"   Locales: {', '.join(locales)}")
    print(f"   History: {days} days")
    
    stats = await ingester.ingest_batch_history(
        app_id=app_id,
        locales=locales, 
        days_back=days
    )
    
    print(f"✅ Ingested {stats.total_reviews} reviews")
    return stats


async def analyze_anomalies(app_id: str):
    """Run anomaly detection analysis."""
    from pavel.clustering.smart_detection_pipeline import SmartDetectionPipeline
    from pavel.embeddings.embedding_pipeline import EmbeddingPipeline, PipelineConfig
    
    print(f"🧠 Running anomaly analysis for {app_id}")
    
    # Setup embedding pipeline
    embedding_config = PipelineConfig(
        embedding_model="intfloat/multilingual-e5-small"
    )
    embedding_pipeline = EmbeddingPipeline(embedding_config)
    
    # Setup smart detection
    detection_pipeline = SmartDetectionPipeline(
        embedding_pipeline=embedding_pipeline
    )
    
    # TODO: Fetch reviews from MongoDB and analyze
    print("🚧 Analysis functionality coming soon...")
    

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="PAVEL - Problem & Anomaly Vector Embedding Locator")
    parser.add_argument("--version", action="version", version="PAVEL 0.6.0")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest reviews from Google Play")
    ingest_parser.add_argument("--app-id", default=get_default_app_id(), 
                              help="App ID to analyze")
    ingest_parser.add_argument("--locales", nargs="+", default=["en", "ru"], 
                              help="Locales to fetch")
    ingest_parser.add_argument("--days", type=int, default=90,
                              help="Days of history to fetch")
    
    # Analyze command  
    analyze_parser = subparsers.add_parser("analyze", help="Run anomaly detection")
    analyze_parser.add_argument("--app-id", default=get_default_app_id(),
                               help="App ID to analyze")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "ingest":
            asyncio.run(ingest_reviews(args.app_id, args.locales, args.days))
        elif args.command == "analyze":
            asyncio.run(analyze_anomalies(args.app_id))
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
    except Exception as e:
        logger.error(f"Command failed: {e}")
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    main()