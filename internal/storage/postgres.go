package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Review represents a user review from an app store
type Review struct {
	ID         int64
	Text       string
	Rating     int
	ThumbsUp   int
	Region     string
	AppID      string
	AppVersion string
	Source     string
	ExternalID string
	URL        string
	UserName   string
	UserImage  string
	ReplyText  string
	ReplyDate  time.Time
	Criterias  map[string]int // e.g. {"Graphics": 4, "Gameplay": 5}
	CreatedAt  time.Time
	IngestedAt time.Time
	Embedding  []float32
}

// Storage handles database operations
type Storage struct {
	pool *pgxpool.Pool
}

// New creates a new Storage instance
func New(ctx context.Context, databaseURL string) (*Storage, error) {
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	// Connection pool settings
	config.MaxConns = 10
	config.MinConns = 2
	config.MaxConnLifetime = time.Hour
	config.MaxConnIdleTime = 30 * time.Minute

	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("create pool: %w", err)
	}

	// Test connection
	if err := pool.Ping(ctx); err != nil {
		return nil, fmt.Errorf("ping: %w", err)
	}

	return &Storage{pool: pool}, nil
}

// Close closes the database connection pool
func (s *Storage) Close() {
	s.pool.Close()
}

// SaveReviews saves multiple reviews to the database
// Returns the number of new reviews inserted (skips duplicates)
func (s *Storage) SaveReviews(ctx context.Context, reviews []Review) (int, error) {
	if len(reviews) == 0 {
		return 0, nil
	}

	// Use a batch for efficiency
	batch := &pgx.Batch{}

	query := `
		INSERT INTO reviews (
			text, rating, thumbs_up, region, app_id, app_version,
			source, external_id, url, user_name, user_image,
			reply_text, reply_date, criterias, created_at
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
		ON CONFLICT (source, external_id) DO UPDATE SET
			text = EXCLUDED.text,
			rating = EXCLUDED.rating,
			thumbs_up = EXCLUDED.thumbs_up,
			app_version = EXCLUDED.app_version,
			reply_text = EXCLUDED.reply_text,
			reply_date = EXCLUDED.reply_date
	`

	for _, r := range reviews {
		// Convert criterias map to JSON
		var criteriasJSON []byte
		if r.Criterias != nil {
			criteriasJSON, _ = json.Marshal(r.Criterias)
		}

		// Handle zero reply date
		var replyDate *time.Time
		if !r.ReplyDate.IsZero() {
			replyDate = &r.ReplyDate
		}

		batch.Queue(query,
			r.Text, r.Rating, r.ThumbsUp, r.Region, r.AppID, r.AppVersion,
			r.Source, r.ExternalID, r.URL, r.UserName, r.UserImage,
			r.ReplyText, replyDate, criteriasJSON, r.CreatedAt,
		)
	}

	results := s.pool.SendBatch(ctx, batch)
	defer results.Close()

	inserted := 0
	for range reviews {
		tag, err := results.Exec()
		if err != nil {
			return inserted, fmt.Errorf("exec: %w", err)
		}
		inserted += int(tag.RowsAffected())
	}

	return inserted, nil
}

// GetReviewCount returns the total number of reviews for an app
func (s *Storage) GetReviewCount(ctx context.Context, appID string) (int64, error) {
	var count int64
	query := `SELECT COUNT(*) FROM reviews WHERE app_id = $1`
	err := s.pool.QueryRow(ctx, query, appID).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("count: %w", err)
	}
	return count, nil
}

// GetTotalReviewCount returns the total number of all reviews
func (s *Storage) GetTotalReviewCount(ctx context.Context) (int64, error) {
	var count int64
	query := `SELECT COUNT(*) FROM reviews`
	err := s.pool.QueryRow(ctx, query).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("count: %w", err)
	}
	return count, nil
}

// GetReviewsWithoutEmbedding returns reviews that don't have embeddings yet
func (s *Storage) GetReviewsWithoutEmbedding(ctx context.Context, limit int) ([]Review, error) {
	query := `
		SELECT id, text, rating, region, app_id, source, external_id, created_at
		FROM reviews
		WHERE embedding IS NULL
		ORDER BY created_at DESC
		LIMIT $1
	`

	rows, err := s.pool.Query(ctx, query, limit)
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()

	var reviews []Review
	for rows.Next() {
		var r Review
		err := rows.Scan(&r.ID, &r.Text, &r.Rating, &r.Region, &r.AppID, &r.Source, &r.ExternalID, &r.CreatedAt)
		if err != nil {
			return nil, fmt.Errorf("scan: %w", err)
		}
		reviews = append(reviews, r)
	}

	return reviews, rows.Err()
}

// LogIngestion logs an ingestion run
func (s *Storage) LogIngestion(ctx context.Context, appID string, reviewsAdded int, status string, errMsg string) error {
	query := `
		INSERT INTO ingestion_log (app_id, finished_at, reviews_added, status, error)
		VALUES ($1, NOW(), $2, $3, $4)
	`
	_, err := s.pool.Exec(ctx, query, appID, reviewsAdded, status, errMsg)
	return err
}

// GetLatestReviewDate returns the date of the most recent review for an app
func (s *Storage) GetLatestReviewDate(ctx context.Context, appID string) (time.Time, error) {
	var t time.Time
	query := `SELECT COALESCE(MAX(created_at), '1970-01-01'::timestamptz) FROM reviews WHERE app_id = $1`
	err := s.pool.QueryRow(ctx, query, appID).Scan(&t)
	return t, err
}

// SaveEmbedding saves an embedding vector for a review
func (s *Storage) SaveEmbedding(ctx context.Context, reviewID int64, embedding []float32) error {
	// Convert to pgvector format: [0.1, 0.2, 0.3, ...]
	vecStr := "["
	for i, v := range embedding {
		if i > 0 {
			vecStr += ","
		}
		vecStr += fmt.Sprintf("%f", v)
	}
	vecStr += "]"

	query := `UPDATE reviews SET embedding = $1::vector WHERE id = $2`
	_, err := s.pool.Exec(ctx, query, vecStr, reviewID)
	return err
}
