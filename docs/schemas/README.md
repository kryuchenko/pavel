# PAVEL MongoDB Schemas

## Schema Files
- `mongodb_schema.js` - Complete MongoDB schema with all collections and indexes
- `schema_validation.js` - MongoDB validation rules for data integrity

## Usage
```bash
# Apply schema to MongoDB
mongosh < docs/schemas/mongodb_schema.js

# Apply validation rules
mongosh < docs/schemas/schema_validation.js
```

## Collections
1. **reviews** - Main review data with all Google Play fields
2. **clusters** - Bug clusters with 2W→2W tracking
3. **app_metadata** - Application configuration
4. **processing_logs** - Pipeline audit trail
5. **alerts** - Monitoring and notifications
6. **embeddings_cache** - Vector embedding cache

## Key Features
- Complete Google Play field preservation
- Anti-duplication via composite keys
- Optimized indexes for performance
- MongoDB 8 Vector Search ready