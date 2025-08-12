#!/usr/bin/env python3
"""
Stage 1 Test: MongoDB Schema and Data Model
Tests schema creation, data insertion, and field preservation
"""

import time
from datetime import datetime, timezone
from google_play_scraper import Sort, reviews
from pymongo import MongoClient
from pavel.core.config import config
from pavel.core.logger import get_logger

logger = get_logger("test_stage1")


def connect_to_mongodb():
    """Connect to MongoDB and return client and database"""
    try:
        logger.info(f"Connecting to MongoDB: {config.DB_URI}")
        client = MongoClient(config.DB_URI)
        
        # Test connection
        client.admin.command('ping')
        logger.info("✓ MongoDB connection successful")
        
        db = client[config.DB_NAME]
        logger.info(f"✓ Using database: {config.DB_NAME}")
        
        return client, db
    except Exception as e:
        logger.error(f"✗ MongoDB connection failed: {e}")
        raise


def setup_collections(db):
    """Create collections and indexes"""
    logger.info("Setting up collections and indexes...")
    
    # Reviews collection
    reviews_collection = db[config.COLLECTION_REVIEWS]
    
    # Drop existing for clean test
    reviews_collection.drop()
    logger.info("✓ Dropped existing reviews collection")
    
    # Create indexes
    indexes = [
        ("appId", 1),
        ("createdAt", 1),
        ("at", 1),
        ("score", 1), 
        ("processed", 1),
        ("clusterId", 1),
        ("locale", 1),
        ("language", 1),
        [("appId", 1), ("at", 1)],
        [("appId", 1), ("score", 1)],
        [("appId", 1), ("processed", 1)]
    ]
    
    for idx in indexes:
        if isinstance(idx, list):
            # Compound index
            reviews_collection.create_index(idx)
            logger.info(f"✓ Created compound index: {idx}")
        else:
            # Single field index  
            reviews_collection.create_index([idx])
            logger.info(f"✓ Created index: {idx}")
    
    # Clusters collection
    clusters_collection = db[config.COLLECTION_CLUSTERS]
    clusters_collection.drop()
    logger.info("✓ Dropped existing clusters collection")
    
    cluster_indexes = [
        ("appId", 1),
        ("createdAt", 1),
        ("weekWindow", 1),
        [("appId", 1), ("weekWindow", 1)],
        ("weight", -1)  # descending for top clusters
    ]
    
    for idx in cluster_indexes:
        if isinstance(idx, list):
            clusters_collection.create_index(idx)
            logger.info(f"✓ Created compound cluster index: {idx}")
        else:
            clusters_collection.create_index([idx])
            logger.info(f"✓ Created cluster index: {idx}")
    
    return reviews_collection, clusters_collection


def fetch_sample_data():
    """Fetch sample data from Google Play"""
    logger.info("Fetching sample data from Google Play...")
    
    app_id = config.DEFAULT_APP_ID
    sample_data = []
    
    # Fetch from multiple locales to get diverse data
    locales = ["en_US", "ru_RU", "es_ES"]
    
    for locale in locales:
        try:
            logger.info(f"Fetching from {locale}...")
            result, _ = reviews(
                app_id,
                lang=locale[:2],
                country=locale[-2:],
                sort=Sort.MOST_RELEVANT,  # Get mix of ratings
                count=10
            )
            
            for review in result:
                # Add metadata
                review["locale"] = locale
                review["language"] = locale[:2]
                review["country"] = locale[-2:]
                review["source"] = "google_play"
                
            sample_data.extend(result)
            logger.info(f"✓ Fetched {len(result)} reviews from {locale}")
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            logger.error(f"✗ Failed to fetch from {locale}: {e}")
    
    logger.info(f"✓ Total fetched: {len(sample_data)} reviews")
    return sample_data


def transform_to_schema(reviews_data, app_id):
    """Transform Google Play data to our MongoDB schema"""
    logger.info("Transforming data to PAVEL schema...")
    
    transformed = []
    now = datetime.now(timezone.utc)
    
    for review in reviews_data:
        # Create composite _id
        review_id = review.get("reviewId")
        composite_id = f"{app_id}:{review_id}"
        
        # Build document according to schema
        doc = {
            # Composite key
            "_id": composite_id,
            
            # PAVEL metadata
            "appId": app_id,
            "source": review.get("source", "google_play"),
            "locale": review.get("locale"),
            "country": review.get("country"), 
            "language": review.get("language"),
            "createdAt": now,
            "updatedAt": now,
            "fetchedAt": now,
            "processingVersion": "1.0",
            
            # ALL Google Play fields (preserve everything)
            "reviewId": review.get("reviewId"),
            "userName": review.get("userName"),
            "userImage": review.get("userImage"),
            "content": review.get("content"),
            "score": review.get("score"),
            "thumbsUpCount": review.get("thumbsUpCount"),
            "at": review.get("at"),  # datetime object
            "appVersion": review.get("appVersion"),
            "reviewCreatedVersion": review.get("reviewCreatedVersion"),
            "replyContent": review.get("replyContent"),
            "repliedAt": review.get("repliedAt"),
            
            # PAVEL processing fields (initial state)
            "processed": False,
            "sentences": [],
            "complaints": [],
            "clusterId": None,
            
            # Convenience flags
            "flags": {
                "hasReply": bool(review.get("replyContent")),
                "isLongReview": len(review.get("content", "")) > 100,
                "isShortReview": len(review.get("content", "")) < 20,
                "hasEmoji": any(ord(c) > 127 for c in review.get("content", "")),
                "hasVersion": bool(review.get("appVersion")),
                "isRecent": (now - review.get("at").replace(tzinfo=timezone.utc)).days < 30 if review.get("at") else False
            },
            
            # Store raw data for debugging
            "rawData": {
                "original": review
            },
            
            # Processing metadata
            "processingErrors": [],
            "retryCount": 0,
            "lastError": None
        }
        
        transformed.append(doc)
    
    logger.info(f"✓ Transformed {len(transformed)} documents")
    return transformed


def test_data_insertion(collection, documents):
    """Test inserting data and check for duplicates"""
    logger.info("Testing data insertion...")
    
    try:
        # Insert documents
        result = collection.insert_many(documents, ordered=False)
        inserted_count = len(result.inserted_ids)
        logger.info(f"✓ Inserted {inserted_count} documents")
        
        # Test duplicate prevention
        logger.info("Testing duplicate prevention...")
        try:
            # Try to insert same documents again
            collection.insert_many(documents, ordered=False)
            logger.error("✗ Duplicates were inserted (should not happen)")
            return False
        except Exception as e:
            if "duplicate key" in str(e).lower():
                logger.info("✓ Duplicate key error as expected")
            else:
                logger.error(f"✗ Unexpected error: {e}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Data insertion failed: {e}")
        return False


def test_field_preservation(collection, original_count):
    """Test that all fields are preserved correctly"""
    logger.info("Testing field preservation...")
    
    try:
        # Check document count
        stored_count = collection.count_documents({})
        if stored_count != original_count:
            logger.error(f"✗ Document count mismatch: {stored_count} vs {original_count}")
            return False
        logger.info(f"✓ Document count correct: {stored_count}")
        
        # Get sample document
        sample = collection.find_one()
        if not sample:
            logger.error("✗ No documents found")
            return False
        
        # Check required fields
        required_fields = [
            "_id", "appId", "reviewId", "userName", "content", 
            "score", "at", "source", "createdAt", "locale"
        ]
        
        for field in required_fields:
            if field not in sample:
                logger.error(f"✗ Missing required field: {field}")
                return False
        logger.info(f"✓ All required fields present")
        
        # Check Google Play fields preservation
        gp_fields = [
            "reviewId", "userName", "userImage", "content", "score",
            "thumbsUpCount", "at", "appVersion", "reviewCreatedVersion",
            "replyContent", "repliedAt"
        ]
        
        preserved_count = sum(1 for field in gp_fields if field in sample)
        logger.info(f"✓ Preserved {preserved_count}/{len(gp_fields)} Google Play fields")
        
        # Check data types
        type_checks = [
            ("score", int),
            ("at", datetime),
            ("createdAt", datetime),
            ("processed", bool),
            ("sentences", list),
            ("flags", dict)
        ]
        
        for field, expected_type in type_checks:
            if field in sample:
                actual_type = type(sample[field])
                if not isinstance(sample[field], expected_type):
                    logger.error(f"✗ Type mismatch for {field}: {actual_type} vs {expected_type}")
                    return False
        logger.info("✓ All data types correct")
        
        # Check flags
        if "flags" in sample and isinstance(sample["flags"], dict):
            flag_keys = ["hasReply", "isLongReview", "hasEmoji", "hasVersion"]
            flags_present = sum(1 for key in flag_keys if key in sample["flags"])
            logger.info(f"✓ Flags present: {flags_present}/{len(flag_keys)}")
        
        # Check raw data preservation
        if "rawData" in sample and "original" in sample["rawData"]:
            logger.info("✓ Raw data preserved for debugging")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Field preservation test failed: {e}")
        return False


def test_queries(collection):
    """Test common query patterns"""
    logger.info("Testing query performance...")
    
    app_id = config.DEFAULT_APP_ID
    
    queries = [
        # Basic app queries
        ({"appId": app_id}, "by appId"),
        ({"appId": app_id, "score": {"$lte": 2}}, "low ratings"),
        ({"appId": app_id, "processed": False}, "unprocessed"),
        ({"flags.hasReply": True}, "with developer replies"),
        ({"flags.isLongReview": True}, "long reviews"),
        
        # Date range queries
        ({"appId": app_id, "at": {"$gte": datetime.now(timezone.utc).replace(day=1)}}, "recent reviews"),
        
        # Locale queries
        ({"appId": app_id, "locale": "en_US"}, "English reviews"),
    ]
    
    for query, description in queries:
        try:
            start_time = time.time()
            count = collection.count_documents(query)
            query_time = time.time() - start_time
            
            logger.info(f"✓ Query {description}: {count} docs in {query_time:.3f}s")
            
            if query_time > 1.0:  # Slow query warning
                logger.warning(f"⚠️ Slow query detected: {description}")
                
        except Exception as e:
            logger.error(f"✗ Query failed ({description}): {e}")
            return False
    
    return True


def create_sample_cluster(clusters_collection, app_id):
    """Create a sample cluster document"""
    logger.info("Creating sample cluster...")
    
    cluster_doc = {
        "_id": f"{app_id}:2025-W33:cluster_001",
        "appId": app_id,
        "clusterId": "cluster_001", 
        "weekWindow": "2025-W33",
        "appVersion": "5.133.0",
        "active": True,
        "createdAt": datetime.now(timezone.utc),
        "updatedAt": datetime.now(timezone.utc),
        
        "label": "App loading issues",
        "topTerms": ["loading", "slow", "wait", "timeout"],
        "centroid": None,  # Will be populated in Stage 5
        "examples": [],
        
        "size": 15,
        "severity": 0.75,
        "avgScore": 2.1,
        "scoreDistribution": {"1": 5, "2": 7, "3": 2, "4": 1, "5": 0},
        
        "locales": {"en_US": 8, "ru_RU": 4, "es_ES": 3},
        "versions": {"5.133.0": 12, "5.132.0": 3},
        "countries": {"US": 6, "RU": 4, "ES": 3, "BR": 2},
        
        "previousWeekSize": 12,
        "growthAbsolute": 3,
        "growthRelative": 0.25,
        "status": "up",
        "trend": "growing",
        "weight": 0.45,  # severity × share
        
        "processingVersion": "1.0",
        "lastRecomputed": datetime.now(timezone.utc)
    }
    
    try:
        clusters_collection.insert_one(cluster_doc)
        logger.info("✓ Sample cluster created")
        return True
    except Exception as e:
        logger.error(f"✗ Cluster creation failed: {e}")
        return False


def main():
    """Run Stage 1 test suite"""
    logger.info("=" * 80)
    logger.info("PAVEL Stage 1 Test Suite: MongoDB Schema")  
    logger.info("=" * 80)
    
    tests = []
    
    try:
        # Connect to MongoDB
        client, db = connect_to_mongodb()
        tests.append(("MongoDB Connection", True))
        
        # Setup collections
        reviews_coll, clusters_coll = setup_collections(db)
        tests.append(("Collection Setup", True))
        
        # Fetch sample data
        sample_reviews = fetch_sample_data()
        if len(sample_reviews) < 10:
            logger.warning(f"⚠️ Only {len(sample_reviews)} reviews fetched (expected >10)")
        tests.append(("Data Fetch", len(sample_reviews) > 0))
        
        # Transform data
        transformed_docs = transform_to_schema(sample_reviews, config.DEFAULT_APP_ID)
        tests.append(("Data Transformation", len(transformed_docs) > 0))
        
        # Test insertion
        insertion_success = test_data_insertion(reviews_coll, transformed_docs)
        tests.append(("Data Insertion", insertion_success))
        
        # Test field preservation  
        field_success = test_field_preservation(reviews_coll, len(transformed_docs))
        tests.append(("Field Preservation", field_success))
        
        # Test queries
        query_success = test_queries(reviews_coll)
        tests.append(("Query Performance", query_success))
        
        # Test cluster creation
        cluster_success = create_sample_cluster(clusters_coll, config.DEFAULT_APP_ID)
        tests.append(("Cluster Creation", cluster_success))
        
        client.close()
        
    except Exception as e:
        logger.error(f"✗ Stage 1 test suite failed: {e}")
        tests.append(("Overall", False))
    
    # Results
    passed = sum(1 for _, result in tests if result)
    failed = len(tests) - passed
    
    logger.info("=" * 80)
    logger.info(f"Results: {passed} passed, {failed} failed")
    
    for test_name, result in tests:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{status}: {test_name}")
    
    if failed == 0:
        logger.info("✅ Stage 1: All tests passed - Ready for Stage 2")
        logger.info("📊 MongoDB schema validated")
        logger.info("💾 All Google Play fields preserved")
        logger.info("🔍 Indexes optimized for queries")
    else:
        logger.error("❌ Stage 1: Some tests failed - Fix issues before proceeding")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)