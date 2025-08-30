// MongoDB initialization script for PAVEL
// Creates database, user, and collections with indexes

db = db.getSiblingDB('gp');

// Create user for the application
db.createUser({
  user: 'pavel',
  pwd: 'pavel123',
  roles: [
    {
      role: 'readWrite',
      db: 'gp'
    }
  ]
});

// Create collections with optimal indexes
db.createCollection('com.nianticlabs.pokemongo');
db.createCollection('clusters');
db.createCollection('embeddings');

// Create indexes for Pokemon GO collection
db['com.nianticlabs.pokemongo'].createIndex({ "reviewId": 1 }, { unique: true });
db['com.nianticlabs.pokemongo'].createIndex({ "appId": 1 });
db['com.nianticlabs.pokemongo'].createIndex({ "locale": 1 });
db['com.nianticlabs.pokemongo'].createIndex({ "score": 1 });
db['com.nianticlabs.pokemongo'].createIndex({ "at": 1 });
db['com.nianticlabs.pokemongo'].createIndex({ "embedding.model": 1 });
db['com.nianticlabs.pokemongo'].createIndex({ "embedding.created_at": 1 });

// Compound indexes for common queries
db['com.nianticlabs.pokemongo'].createIndex({ "locale": 1, "score": 1 });
db['com.nianticlabs.pokemongo'].createIndex({ "appId": 1, "at": 1 });

// Text search index for content
db['com.nianticlabs.pokemongo'].createIndex({ "content": "text" });

print('✅ PAVEL database initialized successfully');
print('📊 Collections created: com.nianticlabs.pokemongo, clusters, embeddings');
print('🔍 Indexes created for optimal query performance');
print('👤 User "pavel" created with readWrite permissions');