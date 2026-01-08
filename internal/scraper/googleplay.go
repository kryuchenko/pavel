package scraper

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	googleplayscraper "github.com/kryuchenko/google-play-scraper"
	"github.com/kryuchenko/pavel/internal/storage"
)

// GooglePlay scrapes reviews from Google Play Store
type GooglePlay struct {
	client *googleplayscraper.Client
	logger *slog.Logger
}

// New creates a new GooglePlay scraper
func New(logger *slog.Logger) *GooglePlay {
	client := googleplayscraper.NewClient()
	return &GooglePlay{
		client: client,
		logger: logger,
	}
}

// FetchReviews fetches reviews for an app from specified regions
func (g *GooglePlay) FetchReviews(ctx context.Context, appID string, regions []string, limit int) ([]storage.Review, error) {
	var allReviews []storage.Review

	for _, region := range regions {
		g.logger.Info("fetching reviews",
			"app_id", appID,
			"region", region,
			"limit", limit,
		)

		reviews, err := g.fetchRegionReviews(ctx, appID, region, limit)
		if err != nil {
			g.logger.Error("failed to fetch reviews",
				"app_id", appID,
				"region", region,
				"error", err,
			)
			continue // Don't fail completely, try other regions
		}

		g.logger.Info("fetched reviews",
			"app_id", appID,
			"region", region,
			"count", len(reviews),
		)

		allReviews = append(allReviews, reviews...)
	}

	return allReviews, nil
}

func (g *GooglePlay) fetchRegionReviews(ctx context.Context, appID, region string, limit int) ([]storage.Review, error) {
	opts := googleplayscraper.ReviewOptions{
		Country: region,
		Sort:    googleplayscraper.SortNewest,
		Count:   limit / 5, // ReviewsComprehensive fetches per rating (5 ratings)
	}

	// Use ReviewsComprehensive to maximize unique reviews (queries each rating separately)
	result, err := g.client.ReviewsComprehensive(ctx, appID, opts)
	if err != nil {
		return nil, fmt.Errorf("fetch reviews: %w", err)
	}

	reviews := make([]storage.Review, 0, len(result))
	for _, r := range result {
		// Skip empty reviews
		if r.Text == "" {
			continue
		}

		// Convert criterias slice to map
		var criterias map[string]int
		if len(r.Criterias) > 0 {
			criterias = make(map[string]int)
			for _, c := range r.Criterias {
				criterias[c.Name] = c.Rating
			}
		}

		review := storage.Review{
			Text:       r.Text,
			Rating:     r.Score,
			ThumbsUp:   r.ThumbsUp,
			Region:     region,
			AppID:      appID,
			AppVersion: r.Version,
			Source:     "google_play",
			ExternalID: r.ID,
			URL:        r.URL,
			UserName:   r.UserName,
			UserImage:  r.UserImage,
			ReplyText:  r.ReplyText,
			ReplyDate:  r.ReplyDate,
			Criterias:  criterias,
			CreatedAt:  r.Date,
		}

		reviews = append(reviews, review)
	}

	return reviews, nil
}

// FetchAllReviews fetches all available reviews (uses pagination)
func (g *GooglePlay) FetchAllReviews(ctx context.Context, appID string, regions []string) ([]storage.Review, error) {
	var allReviews []storage.Review

	for _, region := range regions {
		g.logger.Info("fetching all reviews",
			"app_id", appID,
			"region", region,
		)

		opts := googleplayscraper.ReviewOptions{
			Country: region,
			Sort:    googleplayscraper.SortNewest,
		}

		result, err := g.client.ReviewsAll(ctx, appID, opts)
		if err != nil {
			g.logger.Error("failed to fetch all reviews",
				"app_id", appID,
				"region", region,
				"error", err,
			)
			continue
		}

		for _, r := range result {
			if r.Text == "" {
				continue
			}

			// Convert criterias slice to map
			var criterias map[string]int
			if len(r.Criterias) > 0 {
				criterias = make(map[string]int)
				for _, c := range r.Criterias {
					criterias[c.Name] = c.Rating
				}
			}

			review := storage.Review{
				Text:       r.Text,
				Rating:     r.Score,
				ThumbsUp:   r.ThumbsUp,
				Region:     region,
				AppID:      appID,
				AppVersion: r.Version,
				Source:     "google_play",
				ExternalID: r.ID,
				URL:        r.URL,
				UserName:   r.UserName,
				UserImage:  r.UserImage,
				ReplyText:  r.ReplyText,
				ReplyDate:  r.ReplyDate,
				Criterias:  criterias,
				CreatedAt:  r.Date,
			}
			allReviews = append(allReviews, review)
		}

		g.logger.Info("fetched all reviews",
			"app_id", appID,
			"region", region,
			"count", len(result),
		)

		// Small delay between regions to avoid rate limiting
		time.Sleep(500 * time.Millisecond)
	}

	return allReviews, nil
}

// GetAppInfo returns basic app information
func (g *GooglePlay) GetAppInfo(ctx context.Context, appID string) (string, error) {
	app, err := g.client.App(ctx, appID, googleplayscraper.AppOptions{})
	if err != nil {
		return "", fmt.Errorf("get app: %w", err)
	}
	return app.Title, nil
}
