#!/usr/bin/env python3
"""
Explore all possible fields from Google Play scraper
Check what data is available for schema design
"""

import json
from google_play_scraper import Sort, reviews
from pavel.core.config import config
from pavel.core.logger import get_logger

logger = get_logger("explore_fields")


def analyze_review_fields():
    """Analyze all fields returned by reviews() function"""
    logger.info("Analyzing Google Play review fields...")
    
    app_id = config.DEFAULT_APP_ID
    locales = ["en_US", "ru_RU", "es_ES", "pt_BR", "id_ID"]
    
    all_fields = set()
    field_examples = {}
    field_types = {}
    field_presence = {}
    
    for locale in locales[:3]:  # Check first 3 locales
        try:
            logger.info(f"Fetching from {locale}...")
            
            result, _ = reviews(
                app_id,
                lang=locale[:2],
                country=locale[-2:],
                sort=Sort.NEWEST,
                count=20  # Get more samples
            )
            
            logger.info(f"Got {len(result)} reviews from {locale}")
            
            for review in result:
                # Collect all field names
                for field, value in review.items():
                    all_fields.add(field)
                    
                    # Track field presence
                    if field not in field_presence:
                        field_presence[field] = {"total": 0, "non_null": 0, "non_empty": 0}
                    
                    field_presence[field]["total"] += 1
                    
                    if value is not None:
                        field_presence[field]["non_null"] += 1
                        
                        # Track field types and examples
                        field_type = type(value).__name__
                        if field not in field_types:
                            field_types[field] = set()
                        field_types[field].add(field_type)
                        
                        # Store examples (only if not empty)
                        if value != "" and value != [] and field not in field_examples:
                            if isinstance(value, str) and len(value) > 100:
                                field_examples[field] = value[:100] + "..."
                            else:
                                field_examples[field] = value
                        
                        # Check if non-empty
                        if (isinstance(value, str) and value.strip()) or \
                           (isinstance(value, (int, float)) and value != 0) or \
                           (isinstance(value, list) and value) or \
                           (isinstance(value, dict) and value) or \
                           (not isinstance(value, (str, list, dict))):
                            field_presence[field]["non_empty"] += 1
                            
        except Exception as e:
            logger.error(f"Failed to fetch from {locale}: {e}")
            continue
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("GOOGLE PLAY SCRAPER FIELD ANALYSIS")
    logger.info("="*80)
    
    logger.info(f"\nTotal unique fields found: {len(all_fields)}")
    logger.info(f"Fields: {sorted(all_fields)}")
    
    logger.info("\n" + "-"*80)
    logger.info("FIELD DETAILS:")
    logger.info("-"*80)
    
    for field in sorted(all_fields):
        presence = field_presence.get(field, {})
        total = presence.get("total", 0)
        non_null = presence.get("non_null", 0)
        non_empty = presence.get("non_empty", 0)
        
        null_pct = ((total - non_null) / total * 100) if total > 0 else 0
        empty_pct = ((total - non_empty) / total * 100) if total > 0 else 0
        
        types_str = ", ".join(sorted(field_types.get(field, {"unknown"})))
        example = field_examples.get(field, "N/A")
        
        logger.info(f"\n📝 {field}:")
        logger.info(f"   Type(s): {types_str}")
        logger.info(f"   Presence: {non_null}/{total} ({100-null_pct:.1f}% non-null)")
        logger.info(f"   Content: {non_empty}/{total} ({100-empty_pct:.1f}% non-empty)")
        logger.info(f"   Example: {example}")
    
    return {
        "fields": sorted(all_fields),
        "field_types": {k: list(v) for k, v in field_types.items()},
        "field_examples": field_examples,
        "field_presence": field_presence
    }


def check_edge_cases():
    """Check edge cases and special scenarios"""
    logger.info("\n" + "="*80)
    logger.info("CHECKING EDGE CASES")
    logger.info("="*80)
    
    app_id = config.DEFAULT_APP_ID
    
    # Check different sort orders
    sorts = [Sort.NEWEST, Sort.MOST_RELEVANT, Sort.RATING]
    
    for sort_type in sorts:
        try:
            logger.info(f"\nChecking sort: {sort_type}")
            result, _ = reviews(
                app_id,
                lang="en",
                country="US",
                sort=sort_type,
                count=5
            )
            
            logger.info(f"Got {len(result)} reviews with sort {sort_type}")
            
            # Check score distribution
            scores = [r.get("score", 0) for r in result]
            logger.info(f"Score range: {min(scores)} - {max(scores)}")
            
            # Check content length distribution  
            content_lengths = [len(r.get("content", "")) for r in result]
            logger.info(f"Content length range: {min(content_lengths)} - {max(content_lengths)} chars")
            
            # Check for special fields
            has_reply = sum(1 for r in result if r.get("replyContent"))
            logger.info(f"Reviews with developer reply: {has_reply}/{len(result)}")
            
        except Exception as e:
            logger.error(f"Failed to check sort {sort_type}: {e}")
    
    # Check specific review with known reply (if any)
    try:
        logger.info("\nLooking for reviews with developer replies...")
        result, _ = reviews(
            app_id,
            lang="en", 
            country="US",
            sort=Sort.NEWEST,
            count=50  # Check more reviews for replies
        )
        
        replies = [r for r in result if r.get("replyContent")]
        logger.info(f"Found {len(replies)} reviews with developer replies out of {len(result)}")
        
        if replies:
            sample_reply = replies[0]
            logger.info("\nSample review with reply:")
            logger.info(f"  Review: {sample_reply.get('content', '')[:100]}...")
            logger.info(f"  Reply: {sample_reply.get('replyContent', '')[:100]}...")
            logger.info(f"  Reply date: {sample_reply.get('repliedAt', 'N/A')}")
            
    except Exception as e:
        logger.error(f"Failed to check replies: {e}")


def generate_mongodb_schema():
    """Generate MongoDB schema based on discovered fields"""
    logger.info("\n" + "="*80)
    logger.info("SUGGESTED MONGODB SCHEMA")
    logger.info("="*80)
    
    # Based on common Google Play scraper fields
    schema = {
        "_id": "String (appId:reviewId for uniqueness)",
        "appId": "String (our app identifier)",
        
        # Core review fields
        "reviewId": "String (Google Play review ID)",
        "userName": "String (reviewer name)",
        "userImage": "String (reviewer avatar URL, optional)",
        "content": "String (review text)",
        "score": "Number (1-5 star rating)",
        "thumbsUpCount": "Number (helpful votes, optional)",
        "at": "Date (review creation date)",
        
        # App version info
        "appVersion": "String (app version when reviewed, optional)",
        
        # Developer response
        "replyContent": "String (developer reply text, optional)",
        "repliedAt": "Date (developer reply date, optional)",
        
        # Metadata for processing
        "locale": "String (en_US, ru_RU, etc.)",
        "language": "String (detected language code)",
        "country": "String (user country code)",
        
        # Processing fields (added by PAVEL)
        "createdAt": "Date (when ingested into our system)",
        "updatedAt": "Date (last update)",
        "processed": "Boolean (whether preprocessed)",
        "sentences": "Array of processed sentences",
        "complaints": "Array of complaint sentences with isComplaint flag",
        "clusterId": "String (assigned cluster ID, optional)",
        
        # Additional metadata
        "source": "String (google_play)",
        "fetchedAt": "Date (when fetched from API)",
        "processingVersion": "String (version of processing pipeline)"
    }
    
    logger.info("\nSuggested schema structure:")
    for field, desc in schema.items():
        logger.info(f"  {field:20} : {desc}")
    
    # Indexes
    logger.info("\nSuggested indexes:")
    indexes = [
        "_id (unique)",
        "appId",
        "createdAt",
        "at (review date)",
        "score",
        "processed",
        "clusterId",
        "locale",
        "language",
        "appId + at (compound for time queries)",
        "appId + score (compound for rating queries)"
    ]
    
    for idx in indexes:
        logger.info(f"  - {idx}")
    
    return schema


def main():
    """Run field exploration"""
    logger.info("GOOGLE PLAY SCRAPER FIELD EXPLORATION")
    logger.info("Analyzing inDrive app: sinet.startup.inDriver")
    
    try:
        # Analyze fields
        field_data = analyze_review_fields()
        
        # Check edge cases
        check_edge_cases()
        
        # Generate schema
        schema = generate_mongodb_schema()
        
        # Save results
        results = {
            "app_id": config.DEFAULT_APP_ID,
            "analysis_timestamp": "2025-08-12",
            "field_data": field_data,
            "suggested_schema": schema
        }
        
        with open("field_analysis_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"\n✅ Analysis complete!")
        logger.info(f"📄 Results saved to: field_analysis_results.json")
        logger.info(f"🔍 Found {len(field_data['fields'])} unique fields")
        logger.info(f"📊 Ready for MongoDB schema design")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)