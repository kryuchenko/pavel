// MongoDB Schema Validation for PAVEL
// Ensures data integrity and proper field types

// Reviews collection validation schema
db.createCollection("reviews", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["_id", "appId", "reviewId", "userName", "content", "score", "at", "source", "createdAt"],
      properties: {
        _id: {
          bsonType: "string",
          description: "Composite key: appId:reviewId"
        },
        appId: {
          bsonType: "string", 
          description: "Application identifier"
        },
        reviewId: {
          bsonType: "string",
          description: "Google Play review UUID"
        },
        userName: {
          bsonType: "string",
          description: "Reviewer display name"
        },
        userImage: {
          bsonType: ["string", "null"],
          description: "Reviewer avatar URL"
        },
        content: {
          bsonType: "string",
          minLength: 1,
          description: "Review text content"
        },
        score: {
          bsonType: "int",
          minimum: 1,
          maximum: 5,
          description: "Star rating 1-5"
        },
        thumbsUpCount: {
          bsonType: ["int", "null"],
          minimum: 0,
          description: "Helpful votes count"
        },
        at: {
          bsonType: "date",
          description: "Review creation date from Google Play"
        },
        appVersion: {
          bsonType: ["string", "null"],
          description: "App version when reviewed"
        },
        reviewCreatedVersion: {
          bsonType: ["string", "null"],
          description: "Alternative version field"
        },
        replyContent: {
          bsonType: ["string", "null"],
          description: "Developer reply content"
        },
        repliedAt: {
          bsonType: ["date", "null"],
          description: "Developer reply timestamp"
        },
        source: {
          bsonType: "string",
          enum: ["google_play", "app_store", "manual"],
          description: "Data source"
        },
        locale: {
          bsonType: ["string", "null"],
          pattern: "^[a-z]{2}_[A-Z]{2}$",
          description: "Locale code like en_US"
        },
        language: {
          bsonType: ["string", "null"],
          pattern: "^[a-z]{2}$",
          description: "Language code like en"
        },
        country: {
          bsonType: ["string", "null"],
          pattern: "^[A-Z]{2}$",
          description: "Country code like US"
        },
        createdAt: {
          bsonType: "date",
          description: "Ingestion timestamp"
        },
        updatedAt: {
          bsonType: ["date", "null"],
          description: "Last update timestamp"
        },
        fetchedAt: {
          bsonType: ["date", "null"],
          description: "API fetch timestamp"
        },
        processed: {
          bsonType: "bool",
          description: "Processing completion status"
        },
        sentences: {
          bsonType: "array",
          description: "Processed sentences array",
          items: {
            bsonType: "object",
            properties: {
              text: { bsonType: "string" },
              normalized: { bsonType: "string" },
              language: { bsonType: ["string", "null"] },
              isComplaint: { bsonType: "bool" },
              embedding: { 
                bsonType: ["array", "null"],
                description: "384-dim embedding vector"
              }
            }
          }
        },
        complaints: {
          bsonType: "array",
          description: "Complaint sentences subset"
        },
        clusterId: {
          bsonType: ["string", "null"],
          description: "Assigned cluster identifier"
        },
        processingVersion: {
          bsonType: ["string", "null"],
          description: "Pipeline version"
        },
        flags: {
          bsonType: ["object", "null"],
          description: "Processing flags",
          properties: {
            hasReply: { bsonType: "bool" },
            isLongReview: { bsonType: "bool" },
            isShortReview: { bsonType: "bool" },
            hasEmoji: { bsonType: "bool" },
            hasVersion: { bsonType: "bool" },
            isRecent: { bsonType: "bool" }
          }
        },
        rawData: {
          bsonType: ["object", "null"],
          description: "Raw scraper response"
        },
        processingErrors: {
          bsonType: "array",
          description: "Processing error log"
        },
        retryCount: {
          bsonType: ["int", "null"],
          minimum: 0,
          description: "Processing retry count"
        },
        lastError: {
          bsonType: ["string", "null"],
          description: "Last processing error"
        }
      }
    }
  }
});

// Clusters collection validation schema
db.createCollection("clusters", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["_id", "appId", "clusterId", "weekWindow", "active", "createdAt"],
      properties: {
        _id: {
          bsonType: "string",
          description: "Composite key: appId:weekWindow:clusterId"
        },
        appId: {
          bsonType: "string",
          description: "Application identifier"
        },
        clusterId: {
          bsonType: "string",
          description: "Cluster identifier"
        },
        weekWindow: {
          bsonType: "string",
          pattern: "^\\d{4}-W\\d{2}$",
          description: "ISO week format: 2025-W33"
        },
        appVersion: {
          bsonType: ["string", "null"],
          description: "Primary app version for cluster"
        },
        active: {
          bsonType: "bool",
          description: "Cluster active status"
        },
        createdAt: {
          bsonType: "date",
          description: "Cluster creation timestamp"
        },
        updatedAt: {
          bsonType: ["date", "null"],
          description: "Last update timestamp"
        },
        label: {
          bsonType: ["string", "null"],
          description: "Auto-generated cluster label"
        },
        topTerms: {
          bsonType: "array",
          description: "Key terms array",
          items: { bsonType: "string" }
        },
        centroid: {
          bsonType: ["array", "null"],
          description: "384-dim centroid embedding"
        },
        examples: {
          bsonType: "array",
          description: "Representative review examples"
        },
        size: {
          bsonType: "int",
          minimum: 1,
          description: "Number of reviews in cluster"
        },
        severity: {
          bsonType: ["double", "null"],
          minimum: 0.0,
          maximum: 1.0,
          description: "Severity score 0-1"
        },
        avgScore: {
          bsonType: ["double", "null"],
          minimum: 1.0,
          maximum: 5.0,
          description: "Average rating"
        },
        scoreDistribution: {
          bsonType: ["object", "null"],
          description: "Rating distribution"
        },
        locales: {
          bsonType: ["object", "null"],
          description: "Locale distribution"
        },
        versions: {
          bsonType: ["object", "null"],
          description: "Version distribution"
        },
        countries: {
          bsonType: ["object", "null"],
          description: "Country distribution"
        },
        previousWeekSize: {
          bsonType: ["int", "null"],
          minimum: 0,
          description: "Size in previous week"
        },
        growthAbsolute: {
          bsonType: ["int", "null"],
          description: "Absolute growth vs previous week"
        },
        growthRelative: {
          bsonType: ["double", "null"],
          description: "Relative growth vs previous week"
        },
        status: {
          bsonType: ["string", "null"],
          enum: ["up", "down", "new", "vanished", "stable"],
          description: "2W→2W comparison status"
        },
        trend: {
          bsonType: ["string", "null"],
          enum: ["growing", "stable", "declining"],
          description: "Trend direction"
        },
        weight: {
          bsonType: ["double", "null"],
          minimum: 0.0,
          description: "Priority weight (severity × share)"
        },
        processingVersion: {
          bsonType: ["string", "null"],
          description: "Pipeline version"
        },
        lastRecomputed: {
          bsonType: ["date", "null"],
          description: "Last recomputation timestamp"
        }
      }
    }
  }
});

// App metadata collection validation
db.createCollection("app_metadata", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["_id", "appId", "trackingEnabled"],
      properties: {
        _id: {
          bsonType: "string",
          description: "Same as appId"
        },
        appId: {
          bsonType: "string",
          description: "Application identifier"
        },
        name: {
          bsonType: ["string", "null"],
          description: "App display name"
        },
        developer: {
          bsonType: ["string", "null"],
          description: "Developer name"
        },
        category: {
          bsonType: ["string", "null"],
          description: "App category"
        },
        trackingEnabled: {
          bsonType: "bool",
          description: "Whether tracking is enabled"
        },
        lastFetched: {
          bsonType: ["date", "null"],
          description: "Last successful fetch"
        },
        totalReviews: {
          bsonType: ["long", "null"],
          minimum: 0,
          description: "Total reviews count"
        },
        avgRating: {
          bsonType: ["double", "null"],
          minimum: 1.0,
          maximum: 5.0,
          description: "Average rating"
        },
        supportedLocales: {
          bsonType: "array",
          description: "Supported locale list",
          items: { 
            bsonType: "string",
            pattern: "^[a-z]{2}_[A-Z]{2}$"
          }
        },
        config: {
          bsonType: ["object", "null"],
          description: "App-specific configuration"
        }
      }
    }
  }
});

console.log("✅ MongoDB validation schemas created");
console.log("🛡️ Data integrity enforced");
console.log("📋 Required fields validated");
console.log("🎯 Field types and ranges checked");