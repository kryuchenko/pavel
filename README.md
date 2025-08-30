<div align="center">
  <img src="pavel.png" alt="PAVEL Logo" width="400"/>
  
  # PAVEL — Problem & Anomaly Vector Embedding Locator

  **Automatic bug detection in Google Play reviews**
</div>

## 🚀 Quick Start

```bash
# Clone and run with one command
git clone <repo>
cd pavel
./deploy.sh

# Done! Now you can search for bugs:
docker exec pavel-app python search_reviews.py "crashes lag" --limit 5
```

## 📊 Features

- **Vector Search** — Semantic search across reviews in 24+ languages
- **Issue Detection** — Automatic identification of technical problems
- **Clustering** — Groups similar bugs and anomalies
- **Trend Monitoring** — Track changes week-over-week
- **REST API** — Integration with your systems

## 🛠 Architecture

```
📱 Google Play Reviews → 🔍 Ingestion → 🧹 Preprocessing → 
→ 🧠 Embeddings (E5-multilingual) → 💾 MongoDB 8 →
→ 🎯 Vector Search & Clustering → 📊 Bug Reports
```

### Key Components

| Component | Description | Status |
|-----------|-------------|--------|
| **Ingestion** | Collect reviews from Google Play (8 languages) | ✅ Ready |
| **Preprocessing** | Text normalization, language detection | ✅ Ready |
| **Embeddings** | Vectorization with multilingual-e5-large (1024D) | ✅ Ready |
| **Vector Search** | MongoDB Atlas + in-memory fallback | ✅ Ready |
| **Clustering** | HDBSCAN for anomaly detection | ✅ Ready |
| **Classification** | ML complaint filter | 🚧 WIP |

## 💻 Usage

### Search Reviews
```bash
# Semantic search
python search_reviews.py "game crashes" --limit 10

# Find issues
python search_reviews.py --issues --locale ru

# Sentiment analysis
python search_reviews.py --sentiment negative --limit 5

# Statistics
python search_reviews.py --stats
```

### Collect New Reviews
```bash
# For specific app
python collect_reviews.py --app-id com.example.app

# With filters
python collect_reviews.py --locale en --days 7
```

### Python API
```python
from pavel.search import VectorSearchEngine
from pavel.clustering import SmartDetectionPipeline

# Vector search
engine = VectorSearchEngine("com.nianticlabs.pokemongo")
results = await engine.search_text("lag issues", limit=10)

# Anomaly detection
pipeline = SmartDetectionPipeline()
anomalies = await pipeline.analyze_reviews(app_id, reviews)
```

## 📁 Project Structure

```
pavel/
├── src/pavel/          # Main package
│   ├── core/          # Configuration and utilities
│   ├── ingestion/     # Google Play data collection
│   ├── embeddings/    # Text vectorization
│   ├── search/        # Vector search
│   └── clustering/    # Anomaly detection
├── docker-compose.yml # Docker configuration
├── deploy.sh         # Deployment script
└── requirements.txt  # Python dependencies
```

## 🔧 Configuration

Main settings in `.env`:

```env
# MongoDB
PAVEL_DB_URI=mongodb://localhost:27017
PAVEL_DB_NAME=gp

# Embedding model
PAVEL_EMBEDDING_MODEL=intfloat/multilingual-e5-large

# Default app
PAVEL_DEFAULT_APP_ID=com.nianticlabs.pokemongo
```

## 📈 Performance

- **Vector search**: ~300ms for 100 documents
- **Embedding generation**: 50+ reviews/sec
- **Clustering**: 1500+ reviews/sec
- **Database**: MongoDB 8 with vector indexes

## 🐳 Docker Deployment

```bash
# Start
./deploy.sh

# Stop
./deploy.sh stop

# Logs
./deploy.sh logs

# Status
./deploy.sh status
```

## 📚 Documentation

- [DEPLOYMENT.md](DEPLOYMENT.md) — Deployment guide
- [VECTOR_SEARCH_SETUP.md](VECTOR_SEARCH_SETUP.md) — Vector search setup
- [DATABASE_MANAGEMENT.md](DATABASE_MANAGEMENT.md) — MongoDB management

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

## 📝 License

MIT

---

**PAVEL** — Finding bugs in reviews automatically 🎯