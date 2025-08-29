# PAVEL Review Collection Usage Guide

## Quick Start

### Basic Usage

Collect reviews using app ID:
```bash
python collect_reviews.py com.nianticlabs.pokemongo
```

Collect reviews using Google Play URL:
```bash
python collect_reviews.py "https://play.google.com/store/apps/details?id=com.nianticlabs.pokemongo&hl=en"
```

## Command Line Options

```bash
python collect_reviews.py <app_input> [options]
```

### Arguments

- `app_input` - Required. Either:
  - App ID: `com.nianticlabs.pokemongo`
  - Google Play URL: `https://play.google.com/store/apps/details?id=com.nianticlabs.pokemongo`

### Options

- `--days <number>` - Number of days to look back (default: 30)
- `--locales <locale1> <locale2>...` - Specific locales to collect (default: all 24 major markets)

## Examples

### 1. Pokemon GO - Full Collection
```bash
python collect_reviews.py com.nianticlabs.pokemongo
```
- Collects last 30 days
- All 24 languages
- Stores in collection: `com.nianticlabs.pokemongo`

### 2. Instagram - Using URL
```bash
python collect_reviews.py "https://play.google.com/store/apps/details?id=com.instagram.android"
```
- Parses URL to extract app ID
- Stores in collection: `com.instagram.android`

### 3. Spotify - Custom Parameters
```bash
python collect_reviews.py com.spotify.music --days 7 --locales en es pt
```
- Last 7 days only
- Only English, Spanish, Portuguese
- Stores in collection: `com.spotify.music`

### 4. TikTok - Quick Collection
```bash
python collect_reviews.py com.zhiliaoapp.musically --days 3 --locales en
```
- Last 3 days
- English only
- Stores in collection: `com.zhiliaoapp.musically`

## Database Structure

Each app gets its own MongoDB collection named exactly as the app_id:

```
Database: gp
├── com.nianticlabs.pokemongo     # Pokemon GO reviews
├── com.instagram.android         # Instagram reviews  
├── com.spotify.music            # Spotify reviews
└── com.zhiliaoapp.musically     # TikTok reviews
```

## Review Schema

Each review document contains:

### Core Fields (from Google Play)
- `reviewId` - Unique review identifier
- `userName` - Reviewer's name
- `userImage` - Avatar URL
- `content` - Review text
- `score` - Rating (1-5)
- `thumbsUpCount` - Number of helpful votes
- `at` - Review date
- `appVersion` - App version when reviewed
- `replyContent` - Developer response (if any)
- `repliedAt` - Developer response date

### PAVEL Metadata
- `appId` - Application identifier
- `locale` - Language/market (en, es, ru, etc.)
- `country` - Country code
- `createdAt` - When ingested
- `updatedAt` - Last update time
- `processingStatus` - Processing state
- `flags` - Analysis flags (hasReply, isPositive, etc.)
- `rawData` - Complete original data

## Supported Locales

Default collection includes 24 major markets:
- **Americas**: en (English), es (Spanish), pt (Portuguese)
- **Europe**: fr (French), de (German), it (Italian), nl (Dutch), sv (Swedish), no (Norwegian), da (Danish), fi (Finnish), pl (Polish), cs (Czech), hu (Hungarian), ro (Romanian), uk (Ukrainian)
- **Asia**: ja (Japanese), ko (Korean), zh (Chinese), hi (Hindi), id (Indonesian), ar (Arabic), tr (Turkish)
- **CIS**: ru (Russian)

## MongoDB Queries

### Check collection status:
```javascript
// In mongosh
use gp
db.getCollectionNames()  // List all apps

// Count reviews for Pokemon GO
db["com.nianticlabs.pokemongo"].countDocuments()

// Get latest reviews
db["com.nianticlabs.pokemongo"].find().sort({at: -1}).limit(5)

// Get reviews by locale
db["com.nianticlabs.pokemongo"].find({locale: "en"}).count()

// Average rating
db["com.nianticlabs.pokemongo"].aggregate([
  {$group: {_id: null, avgScore: {$avg: "$score"}}}
])
```

## Rate Limiting

The collector implements automatic rate limiting:
- Prevents API blocks
- 60-second cooldown after burst limits
- Automatic retry on failures
- Progress logging for monitoring

## Tips

1. **Start small**: Test with `--days 3 --locales en` first
2. **Monitor progress**: Watch the logs for collection status
3. **Incremental updates**: Run daily with `--days 1` for updates
4. **Parallel collections**: Run multiple apps in separate terminals
5. **Check duplicates**: The system prevents duplicate reviews automatically

## Troubleshooting

### No reviews found
- Some apps restrict access by region
- Try different locales
- Check if app ID is correct

### Rate limiting
- Normal behavior, waits automatically
- Don't interrupt, let it continue

### Connection errors
- Ensure MongoDB is running: `brew services list | grep mongodb`
- Check connection: `mongosh --eval "db.adminCommand('ping')"`