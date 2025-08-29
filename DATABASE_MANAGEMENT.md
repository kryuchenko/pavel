# Database Management

## MongoDB Collections Structure

Each app has its own collection named exactly as the app_id:
- `com.nianticlabs.pokemongo` - Pokemon GO reviews
- `com.instagram.android` - Instagram reviews  
- `com.spotify.music` - Spotify reviews

## Common Operations

### List all collections
```bash
mongosh gp --eval "db.getCollectionNames()"
```

### Check collection sizes
```bash
mongosh gp --eval "db.getCollectionNames().forEach(c => print(c + ': ' + db[c].countDocuments() + ' reviews'))"
```

### View collection statistics
```javascript
// In mongosh
use gp
db["com.nianticlabs.pokemongo"].stats()
```

### Delete a collection
```bash
# Delete specific app collection
mongosh gp --eval "db['com.nianticlabs.pokemongo'].drop()"

# Or in mongosh
use gp
db["com.instagram.android"].drop()
```

### Clean up old data
```javascript
// Delete reviews older than 90 days
db["com.nianticlabs.pokemongo"].deleteMany({
  at: {$lt: new Date(Date.now() - 90*24*60*60*1000)}
})

// Delete all reviews for specific locale
db["com.nianticlabs.pokemongo"].deleteMany({locale: "en"})
```

### Export/Backup collection
```bash
# Export to JSON
mongoexport --db=gp --collection=com.nianticlabs.pokemongo --out=pokemon_reviews.json

# Export to CSV
mongoexport --db=gp --collection=com.nianticlabs.pokemongo --type=csv --fields=userName,content,score,at,locale --out=pokemon_reviews.csv
```

### Import data
```bash
# Import from JSON
mongoimport --db=gp --collection=com.nianticlabs.pokemongo --file=pokemon_reviews.json

# Import from CSV
mongoimport --db=gp --collection=com.nianticlabs.pokemongo --type=csv --headerline --file=pokemon_reviews.csv
```

## Storage Optimization

### Compact database
```javascript
// In mongosh
use gp
db.runCommand({compact: "com.nianticlabs.pokemongo"})
```

### Check database size
```javascript
use gp
db.stats()
// Or for specific collection
db["com.nianticlabs.pokemongo"].stats()
```

## Monitoring

### Recent reviews
```javascript
db["com.nianticlabs.pokemongo"]
  .find()
  .sort({at: -1})
  .limit(10)
```

### Reviews by date range
```javascript
db["com.nianticlabs.pokemongo"].find({
  at: {
    $gte: ISODate("2025-08-01"),
    $lte: ISODate("2025-08-31")
  }
}).count()
```

### Language distribution
```javascript
db["com.nianticlabs.pokemongo"].aggregate([
  {$group: {_id: "$locale", count: {$sum: 1}}},
  {$sort: {count: -1}}
])
```

## Best Practices

1. **Regular cleanup**: Delete reviews older than needed (e.g., >180 days)
2. **Backup before deletion**: Always export data before dropping collections
3. **Monitor size**: Check collection sizes regularly
4. **Use indexes**: Already configured in the ingester
5. **Separate collections**: Each app in its own collection for easy management