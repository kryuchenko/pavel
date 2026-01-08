# PAVEL

> ⚠️ **Work in progress**

![PAVEL](pavel.jpg)

**Problem & Anomaly Vector Embedding Locator**

User review analysis system for detecting problems and anomalies using vector embeddings.

## Quick Start

**Requirements:** Go 1.25+, Docker

```bash
# macOS
brew install go docker

# Clone and setup
git clone https://github.com/kryuchenko/pavel
cd pavel
cp .env.example .env    # configure apps to track
make setup              # installs llama.cpp, downloads model (~313MB), starts DB, builds
```

**Run:**

```bash
make run                # start all services

# Or manually:
./bin/pavel ingest      # collect reviews
./bin/pavel embed       # generate embeddings
./bin/pavel serve       # start API server
```

**Configuration** (`.env`):

```bash
APPS=sinet.startup.inDriver    # apps to track (comma-separated)
SCRAPE_INTERVAL=1h             # how often to check for new reviews
SCRAPE_REGIONS=us,gb,de,ru     # regions to scrape
```

## Features

- Collect reviews from Google Play and other sources
- Semantic clustering of similar issues
- Anomaly detection for complaint spikes
- Problem type classification
- Multilingual support

## Stack

- **Go**
- **PostgreSQL + pgvector** — reviews storage + vector search
- **EmbeddingGemma 300M** (768 dim) via llama-server — text embeddings (~74 emb/s)
- **[google-play-scraper](https://github.com/kryuchenko/google-play-scraper)** — Google Play reviews scraping

## Schema

```sql
CREATE EXTENSION vector;

CREATE TABLE reviews (
    id            BIGSERIAL PRIMARY KEY,
    text          TEXT NOT NULL,
    rating        SMALLINT,
    region        VARCHAR(10),
    app_id        VARCHAR(255),
    source        VARCHAR(50),
    external_id   VARCHAR(255),
    created_at    TIMESTAMPTZ,
    embedding     vector(768)
);

CREATE INDEX ON reviews USING hnsw (embedding vector_cosine_ops)
WITH (m = 32, ef_construction = 128);
```

## Why PostgreSQL + pgvector?

Compared alternatives: Qdrant, Weaviate, Milvus, Vespa, SQLite+Faiss.

**pgvector wins for this project:**

- **Single storage** — text, metadata, and embeddings in one place
- **SQL filtering** — `WHERE rating >= 4 AND region = 'US'` + vector search in one query
- **HNSW index** — 95-99% recall, same algorithm as Qdrant/Weaviate
- **ARM compatible** — works on Raspberry Pi, Apple Silicon
- **Scales to ~1M vectors** — enough for review analysis
- **Mature ecosystem** — backups, replication, transactions

**When to switch to Qdrant:**
- 5M+ vectors
- Need <10ms latency
- Heavy filtering workloads

**Tuning for better recall:**

```sql
-- Higher ef_search = better recall, slower search
SET hnsw.ef_search = 200;
```

## Embeddings

Uses [llama.cpp](https://github.com/ggerganov/llama.cpp) server with EmbeddingGemma 300M.

```bash
make llama-start   # starts llama-server on :8090
make embed         # generates embeddings for all reviews
make llama-stop    # stops server
```

**Specs:**
- Model: EmbeddingGemma 300M Q8_0 (313 MB)
- Dimension: 768
- Speed: ~74 emb/s (Apple Silicon)
- Memory: ~200 MB

## Similarity Search

```sql
SELECT id, text, rating, 1 - (embedding <=> $1) AS similarity
FROM reviews
ORDER BY embedding <=> $1
LIMIT 10;
```

## Visualization

Interactive 3D embedding visualization via [Embedding Atlas](https://github.com/apple/embedding-atlas) (Apple, MIT License).

**Setup:**

```bash
pip install embedding-atlas umap-learn psycopg2-binary pandas pyarrow
```

**Export & visualize:**

```bash
# Export embeddings to parquet with UMAP projection
python scripts/export_embeddings.py --app-id com.example.app --limit 100000

# Open interactive visualization in browser
embedding-atlas data/embeddings.parquet
```

**Features:**
- WebGPU/WebGL rendering — handles millions of points
- Automatic clustering & labeling
- Real-time search & nearest neighbors
- Filter by rating, date, region

## Trend Analysis

Track how complaint topics change over time using [BERTopic](https://maartengr.github.io/BERTopic/).

**Setup:**

```bash
pip install bertopic
```

**Usage:**

```python
from bertopic import BERTopic
import numpy as np

# Load from PostgreSQL (texts, timestamps, embeddings)
texts, timestamps, embeddings = load_reviews(db, app_id)

# BERTopic uses pre-computed embeddings
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(texts, embeddings=np.array(embeddings))

# Topics over time
topics_over_time = topic_model.topics_over_time(texts, timestamps, nr_bins=20)

# Interactive visualization (Plotly)
topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
```

**Visualizations:**
- `visualize_topics_over_time()` — trend chart for each topic
- `visualize_topics()` — topic map (distances between topics)
- `visualize_hierarchy()` — topic dendrogram
- `visualize_barchart()` — top words per topic
