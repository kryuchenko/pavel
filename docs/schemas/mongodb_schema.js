// MongoDB Schema for PAVEL - Complete Google Play data preservation
// All possible fields from google-play-scraper + PAVEL processing fields

// Collection: reviews
db.reviews.createIndex({ "_id": 1 }, { unique: true });
db.reviews.createIndex({ "appId": 1 });
db.reviews.createIndex({ "createdAt": 1 });
db.reviews.createIndex({ "at": 1 }); // review date
db.reviews.createIndex({ "score": 1 });
db.reviews.createIndex({ "processed": 1 });
db.reviews.createIndex({ "clusterId": 1 });
db.reviews.createIndex({ "locale": 1 });
db.reviews.createIndex({ "language": 1 });
db.reviews.createIndex({ "appId": 1, "at": 1 }); // compound for time queries
db.reviews.createIndex({ "appId": 1, "score": 1 }); // compound for rating queries
db.reviews.createIndex({ "appId": 1, "processed": 1 }); // compound for processing status
db.reviews.createIndex({ "appId": 1, "locale": 1 }); // compound for locale queries

// Sample document structure with ALL possible Google Play fields preserved
{
  // Unique identifier (appId:reviewId for global uniqueness)
  "_id": "sinet.startup.inDriver:fda324fb-f172-4c13-a432-199f99220650",
  
  // PAVEL metadata
  "appId": "sinet.startup.inDriver", // our app identifier
  "source": "google_play", // data source
  "locale": "en_US", // fetch locale
  "country": "US", // extracted country
  "language": "en", // detected/extracted language
  "createdAt": ISODate("2025-08-12T17:46:50Z"), // ingestion timestamp
  "updatedAt": ISODate("2025-08-12T17:46:50Z"), // last update
  "fetchedAt": ISODate("2025-08-12T17:46:50Z"), // when fetched from GP API
  "processingVersion": "1.0", // pipeline version for schema evolution
  
  // ALL Google Play scraper fields (preserve everything)
  "reviewId": "fda324fb-f172-4c13-a432-199f99220650", // GP review UUID
  "userName": "Prosper Masona", // reviewer display name
  "userImage": "https://play-lh.googleusercontent.com/a-/ALV-UjVTiLZ-ax_eH2KTwssiJhRnqD4F0tzNzbqdP6ALe09ZmPsIEi4", // avatar URL
  "content": "so far so good 👍", // review text (main content)
  "score": 5, // 1-5 star rating
  "thumbsUpCount": 0, // helpful votes count
  "at": ISODate("2025-08-11T19:43:05Z"), // review creation date (GP timestamp)
  "appVersion": "5.133.0", // app version when reviewed (90% present)
  "reviewCreatedVersion": "5.133.0", // duplicate/alternative version field (90% present)
  "replyContent": "Hello! We apologize for any inconvenience...", // developer reply (38% present)
  "repliedAt": ISODate("2024-11-25T06:32:59Z"), // developer reply date (38% present)
  
  // PAVEL processing fields (populated by pipeline stages)
  "processed": false, // whether preprocessing completed
  "sentences": [
    // Array of sentence objects after preprocessing
    {
      "text": "so far so good",
      "normalized": "so far so good",
      "language": "en",
      "isComplaint": false,
      "embedding": null // will be populated in Stage 5
    }
  ],
  "complaints": [
    // Array of complaint sentences (subset of sentences where isComplaint=true)
    // Populated in Stage 4
  ],
  "clusterId": null, // assigned cluster ID (Stage 7)
  
  // Additional metadata for debugging/auditing
  "rawData": {
    // Store complete raw response from google-play-scraper
    // In case new fields are added or for debugging
    "original": {} // complete original object
  },
  
  // Processing flags and metadata
  "flags": {
    "hasReply": true, // convenience flag
    "isLongReview": false, // >100 chars
    "isShortReview": false, // <10 chars
    "hasEmoji": true, // contains emoji
    "hasVersion": true, // has version info
    "isRecent": true // within last 30 days
  },
  
  // Error handling and retry info
  "processingErrors": [], // any errors during processing
  "retryCount": 0, // number of processing retries
  "lastError": null // last processing error
}

// Collection: clusters  
db.clusters.createIndex({ "_id": 1 }, { unique: true });
db.clusters.createIndex({ "appId": 1 });
db.clusters.createIndex({ "createdAt": 1 });
db.clusters.createIndex({ "weekWindow": 1 });
db.clusters.createIndex({ "appId": 1, "weekWindow": 1 });
db.clusters.createIndex({ "appId": 1, "active": 1 });
db.clusters.createIndex({ "weight": -1 }); // descending for top clusters

// Sample cluster document
{
  "_id": "sinet.startup.inDriver:2025-W33:cluster_001",
  "appId": "sinet.startup.inDriver",
  "clusterId": "cluster_001",
  "weekWindow": "2025-W33", // ISO week
  "appVersion": "5.133.0", // primary version for this cluster
  "active": true,
  "createdAt": ISODate("2025-08-12T17:46:50Z"),
  "updatedAt": ISODate("2025-08-12T17:46:50Z"),
  
  // Cluster content
  "label": "App crashes on startup", // auto-generated label
  "topTerms": ["crash", "startup", "loading", "error"], // key terms
  "centroid": [], // embedding centroid (384-dim array)
  "examples": [
    // Top representative reviews
    {
      "reviewId": "...",
      "content": "...",
      "score": 1,
      "distance": 0.05
    }
  ],
  
  // Statistics
  "size": 25, // number of reviews in cluster
  "severity": 0.85, // severity score (0-1)
  "avgScore": 1.2, // average rating
  "scoreDistribution": {
    "1": 20, "2": 3, "3": 1, "4": 1, "5": 0
  },
  
  // Metadata distribution
  "locales": {"en_US": 15, "ru_RU": 5, "es_ES": 5},
  "versions": {"5.133.0": 20, "5.132.0": 5},
  "countries": {"US": 10, "RU": 5, "ES": 5, "BR": 3, "KZ": 2},
  
  // 2W→2W tracking
  "previousWeekSize": 18, // size in previous week
  "growthAbsolute": 7, // +7 reviews
  "growthRelative": 0.39, // +39% growth
  "status": "up", // up/down/new/vanished
  "trend": "growing", // growing/stable/declining
  
  // Processing metadata
  "processingVersion": "1.0",
  "lastRecomputed": ISODate("2025-08-12T17:46:50Z")
}

// Additional collections for advanced features

// Collection: app_metadata (store app-specific info)
db.app_metadata.createIndex({ "appId": 1 }, { unique: true });
{
  "_id": "sinet.startup.inDriver",
  "appId": "sinet.startup.inDriver",
  "name": "inDrive",
  "developer": "SUOL INNOVATIONS LTD",
  "category": "Maps & Navigation",
  "lastFetched": ISODate("2025-08-12T17:46:50Z"),
  "totalReviews": 1250000,
  "avgRating": 4.2,
  "supportedLocales": ["en_US", "ru_RU", "es_ES", "pt_BR", "id_ID", "kk_KZ"],
  "trackingEnabled": true,
  "config": {
    "fetchFrequency": "daily",
    "historyDays": 90,
    "enableClustering": true,
    "enableAlerts": true
  }
}

// Collection: processing_logs (audit trail)
db.processing_logs.createIndex({ "timestamp": 1 });
db.processing_logs.createIndex({ "appId": 1, "stage": 1 });
{
  "_id": ObjectId("..."),
  "appId": "sinet.startup.inDriver",
  "stage": "ingest", // ingest/preprocess/embed/cluster/etc
  "timestamp": ISODate("2025-08-12T17:46:50Z"),
  "status": "success", // success/error/warning
  "message": "Ingested 150 new reviews",
  "details": {
    "reviewsProcessed": 150,
    "reviewsSkipped": 5,
    "errors": 0,
    "duration": 12.5,
    "locale": "en_US"
  },
  "version": "1.0"
}

// Collection: alerts (for monitoring)
db.alerts.createIndex({ "appId": 1, "timestamp": -1 });
db.alerts.createIndex({ "severity": -1, "resolved": 1 });
{
  "_id": ObjectId("..."),
  "appId": "sinet.startup.inDriver", 
  "type": "cluster_spike", // cluster_spike/new_cluster/rating_drop
  "severity": "high", // low/medium/high/critical
  "title": "Crash cluster growing rapidly",
  "message": "Startup crash cluster grew by 150% in last week",
  "clusterId": "cluster_001",
  "timestamp": ISODate("2025-08-12T17:46:50Z"),
  "resolved": false,
  "resolvedAt": null,
  "data": {
    "currentSize": 45,
    "previousSize": 18,
    "growthRate": 1.5,
    "affectedVersions": ["5.133.0"],
    "avgScore": 1.2
  }
}

// Collection: embeddings_cache (for performance)
db.embeddings_cache.createIndex({ "textHash": 1 }, { unique: true });
db.embeddings_cache.createIndex({ "createdAt": 1 }, { expireAfterSeconds: 2592000 }); // 30 days TTL
{
  "_id": ObjectId("..."),
  "textHash": "sha256_hash_of_normalized_text",
  "text": "app crashes on startup",
  "embedding": [], // 384-dim array
  "model": "intfloat/multilingual-e5-base",
  "language": "en",
  "createdAt": ISODate("2025-08-12T17:46:50Z")
}

console.log("✅ MongoDB schema created for PAVEL");
console.log("📊 Collections: reviews, clusters, app_metadata, processing_logs, alerts, embeddings_cache");  
console.log("🔍 All Google Play fields preserved + PAVEL processing fields");
console.log("⚡ Optimized indexes for performance");
console.log("🎯 Default app: sinet.startup.inDriver");