-- PAVEL Schema
-- Problem & Anomaly Vector Embedding Locator

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Reviews table
CREATE TABLE IF NOT EXISTS reviews (
    id            BIGSERIAL PRIMARY KEY,

    -- Review content
    text          TEXT NOT NULL,
    rating        SMALLINT CHECK (rating >= 1 AND rating <= 5),
    thumbs_up     INTEGER DEFAULT 0,

    -- User info
    user_name     VARCHAR(255),
    user_image    TEXT,

    -- Developer reply
    reply_text    TEXT,
    reply_date    TIMESTAMPTZ,

    -- App info
    app_id        VARCHAR(255) NOT NULL,
    app_version   VARCHAR(50),

    -- Metadata
    region        VARCHAR(10),
    source        VARCHAR(50) DEFAULT 'google_play',
    external_id   VARCHAR(255),
    url           TEXT,
    criterias     JSONB,

    -- Timestamps
    created_at    TIMESTAMPTZ,
    ingested_at   TIMESTAMPTZ DEFAULT NOW(),

    -- Embedding
    embedding     vector(768),

    UNIQUE(source, external_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_reviews_app_id ON reviews(app_id);
CREATE INDEX IF NOT EXISTS idx_reviews_created_at ON reviews(created_at);
CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);
CREATE INDEX IF NOT EXISTS idx_reviews_app_version ON reviews(app_version);

-- HNSW index for vector search (created after data is loaded for better performance)
-- CREATE INDEX ON reviews USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 128);

-- Apps table (tracked applications)
CREATE TABLE IF NOT EXISTS apps (
    id            SERIAL PRIMARY KEY,
    app_id        VARCHAR(255) UNIQUE NOT NULL,
    name          VARCHAR(255),
    added_at      TIMESTAMPTZ DEFAULT NOW(),
    last_sync_at  TIMESTAMPTZ,
    enabled       BOOLEAN DEFAULT TRUE
);

-- Ingestion log
CREATE TABLE IF NOT EXISTS ingestion_log (
    id            BIGSERIAL PRIMARY KEY,
    app_id        VARCHAR(255) NOT NULL,
    started_at    TIMESTAMPTZ DEFAULT NOW(),
    finished_at   TIMESTAMPTZ,
    reviews_added INTEGER DEFAULT 0,
    status        VARCHAR(50) DEFAULT 'running',
    error         TEXT
);

-- Helper function: search similar reviews
CREATE OR REPLACE FUNCTION search_similar(
    query_embedding vector(768),
    match_app_id VARCHAR DEFAULT NULL,
    match_count INTEGER DEFAULT 10
)
RETURNS TABLE (
    id BIGINT,
    text TEXT,
    rating SMALLINT,
    app_id VARCHAR,
    created_at TIMESTAMPTZ,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        r.id,
        r.text,
        r.rating,
        r.app_id,
        r.created_at,
        1 - (r.embedding <=> query_embedding) AS similarity
    FROM reviews r
    WHERE r.embedding IS NOT NULL
      AND (match_app_id IS NULL OR r.app_id = match_app_id)
    ORDER BY r.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;
