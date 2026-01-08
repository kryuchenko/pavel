#!/usr/bin/env python3
"""
BERTopic analysis for PAVEL reviews.
Tracks how complaint topics change over time.
"""

import os
import argparse
import json
import random
import requests
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from openai import OpenAI

# Load environment variables
load_dotenv()


def get_db_connection():
    """Connect to PostgreSQL using environment variables."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "pavel"),
        user=os.getenv("DB_USER", "pavel"),
        password=os.getenv("DB_PASSWORD", "pavel"),
    )


def load_reviews(app_id: str = None, limit: int = None, since: str = None) -> pd.DataFrame:
    """Load reviews with embeddings from PostgreSQL."""
    conn = get_db_connection()

    query = """
        SELECT id, text, rating, region, app_id, created_at, embedding
        FROM reviews
        WHERE embedding IS NOT NULL
    """
    params = []

    if app_id:
        query += " AND app_id = %s"
        params.append(app_id)

    if since:
        query += " AND created_at >= %s"
        params.append(since)

    query += " ORDER BY created_at DESC"

    if limit:
        query += " LIMIT %s"
        params.append(limit)

    df = pd.read_sql(query, conn, params=params if params else None)
    conn.close()

    # Parse embeddings from pgvector format
    df["embedding"] = df["embedding"].apply(
        lambda x: np.array([float(v) for v in x.strip("[]").split(",")]) if x else None
    )

    print(f"Loaded {len(df)} reviews with embeddings")
    return df


def train_model(df: pd.DataFrame, nr_topics: int = None) -> BERTopic:
    """Train BERTopic model using pre-computed embeddings."""
    texts = df["text"].tolist()
    embeddings = np.array(df["embedding"].tolist())

    # Configure UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # Configure HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Create BERTopic model
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=nr_topics,
        verbose=True,
    )

    # Fit with pre-computed embeddings
    topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)

    # Add topics to dataframe
    df["topic"] = topics

    return topic_model


def analyze_topics_over_time(
    topic_model: BERTopic,
    df: pd.DataFrame,
    nr_bins: int = 30,
) -> pd.DataFrame:
    """Analyze how topics change over time."""
    texts = df["text"].tolist()
    timestamps = df["created_at"].tolist()

    topics_over_time = topic_model.topics_over_time(
        texts,
        timestamps,
        nr_bins=nr_bins,
    )

    return topics_over_time


STOPWORDS = {
    "the", "to", "is", "for", "and", "no", "not", "in", "of", "it", "this",
    "that", "but", "was", "are", "be", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "must",
    "shall", "can", "need", "dare", "ought", "used", "a", "an", "the",
    "my", "your", "his", "her", "its", "our", "their", "me", "you", "him",
    "us", "them", "i", "he", "she", "we", "they", "what", "which", "who",
    "whom", "whose", "where", "when", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "so", "than", "too", "very", "just", "also", "now", "here",
    "there", "then", "once", "well", "back", "even", "still", "already",
    # Common non-English words that use Latin script
    "ng", "mga", "na", "ang", "sa", "nag", "yung", "pero", "mag", "lang",
    "hai", "ko", "ka", "kam", "ke", "ki", "hi", "koi", "hain", "ha",
    "bhot", "hona", "bhai", "bekar", "nat", "utar", "cancial", "bakwas",
    "aap", "su", "sarve", "virigood", "goood", "vvvv", "sulit", "parson",
    "jhob", "parsan", "sarvise", "realiable", "facelty", "gop", "meron",
    "precio", "overally",
}


def is_good_word(word: str) -> bool:
    """Check if word is meaningful English or Russian."""
    word_lower = word.lower()

    # Skip stopwords
    if word_lower in STOPWORDS:
        return False

    # Skip very short words
    if len(word) < 3:
        return False

    # Check if mostly ASCII (English) or Cyrillic (Russian)
    ascii_count = sum(1 for c in word_lower if 'a' <= c <= 'z')
    cyrillic_count = sum(1 for c in word_lower if '\u0400' <= c <= '\u04ff')

    # Must be mostly one language
    if ascii_count >= len(word) * 0.8:
        return True
    if cyrillic_count >= len(word) * 0.8:
        return True

    return False


def filter_topic_words(topic_model: BERTopic) -> dict:
    """Filter topic words to English/Russian only and create readable names."""
    topic_labels = {}

    for topic_id in topic_model.get_topic_info()["Topic"]:
        if topic_id == -1:
            topic_labels[topic_id] = "Other"
            continue

        words = topic_model.get_topic(topic_id)
        if not words:
            topic_labels[topic_id] = f"Topic {topic_id}"
            continue

        # Filter to meaningful English/Russian words
        filtered = [w for w, _ in words if is_good_word(w)]

        if filtered:
            # Take top 3 filtered words
            topic_labels[topic_id] = ", ".join(filtered[:3])
        else:
            # Fallback to original top word
            topic_labels[topic_id] = words[0][0]

    return topic_labels


def generate_topic_names_gpt(topic_model: BERTopic, df: pd.DataFrame, sample_size: int = 25) -> dict:
    """Generate topic names using GPT based on sample reviews."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set, using keyword-based names")
        return filter_topic_words(topic_model)

    client = OpenAI(api_key=api_key)
    topic_labels = {}

    # Get topics assigned to each document
    topics = topic_model.topics_

    for topic_id in topic_model.get_topic_info()["Topic"]:
        if topic_id == -1:
            topic_labels[topic_id] = "Other"
            continue

        # Get indices of documents in this topic
        topic_indices = [i for i, t in enumerate(topics) if t == topic_id]

        if not topic_indices:
            topic_labels[topic_id] = f"Topic {topic_id}"
            continue

        # Sample random reviews from this topic
        sample_indices = random.sample(topic_indices, min(sample_size, len(topic_indices)))
        sample_reviews = df.iloc[sample_indices]["text"].tolist()

        # Truncate long reviews
        sample_reviews = [r[:500] if len(r) > 500 else r for r in sample_reviews]

        # Create prompt
        reviews_text = "\n---\n".join(sample_reviews)
        prompt = f"""Analyze these app reviews and provide a SHORT topic name (2-5 words in English) that describes the main theme or complaint.

Reviews:
{reviews_text}

Respond with ONLY the topic name, nothing else. Examples of good topic names:
- "Driver cancellation issues"
- "Payment problems"
- "App crashes frequently"
- "GPS location inaccurate"
- "High prices complaints"
- "Positive driver feedback"
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.3,
            )
            topic_name = response.choices[0].message.content.strip().strip('"')
            topic_labels[topic_id] = topic_name
            print(f"  Topic {topic_id}: {topic_name}")
        except Exception as e:
            print(f"  Topic {topic_id}: GPT error - {e}")
            # Fallback to keywords
            words = topic_model.get_topic(topic_id)
            if words:
                filtered = [w for w, _ in words if is_good_word(w)]
                topic_labels[topic_id] = ", ".join(filtered[:3]) if filtered else words[0][0]
            else:
                topic_labels[topic_id] = f"Topic {topic_id}"

    return topic_labels


def save_model(topic_model: BERTopic, path: str = "models/bertopic"):
    """Save trained model for later use."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    topic_model.save(path, serialization="safetensors", save_ctfidf=True)
    print(f"Model saved to {path}")


def load_model(path: str = "models/bertopic") -> BERTopic:
    """Load previously trained model."""
    return BERTopic.load(path)


def main():
    parser = argparse.ArgumentParser(description="BERTopic analysis for PAVEL reviews")
    parser.add_argument("--app-id", help="Filter by app ID")
    parser.add_argument("--limit", type=int, help="Limit number of reviews")
    parser.add_argument("--since", help="Filter reviews since date (YYYY-MM-DD)")
    parser.add_argument("--nr-topics", type=int, help="Reduce to N topics (auto if not set)")
    parser.add_argument("--load-model", help="Load existing model instead of training")
    parser.add_argument("--save-model", default="models/bertopic", help="Path to save model")
    parser.add_argument("--output", default="topics_over_time.html", help="Output HTML file")
    parser.add_argument("--nr-bins", type=int, default=30, help="Number of time bins (default 30)")
    parser.add_argument("--smooth", type=int, default=3, help="Smoothing window size (default 3)")
    parser.add_argument("--gpt-names", action="store_true", help="Use GPT to generate topic names")
    args = parser.parse_args()

    # Load reviews
    print("Loading reviews...")
    df = load_reviews(app_id=args.app_id, limit=args.limit, since=args.since)

    if len(df) == 0:
        print("No reviews found with embeddings")
        return

    # Train or load model
    if args.load_model:
        print(f"Loading model from {args.load_model}...")
        topic_model = load_model(args.load_model)
        # Transform to get topics for our data
        texts = df["text"].tolist()
        embeddings = np.array(df["embedding"].tolist())
        topics, _ = topic_model.transform(texts, embeddings=embeddings)
        df["topic"] = topics
    else:
        print("Training BERTopic model...")
        topic_model = train_model(df, nr_topics=args.nr_topics)

        # Save model
        if args.save_model:
            save_model(topic_model, args.save_model)

    # Print topic info
    print("\n=== Topics ===")
    topic_info = topic_model.get_topic_info()
    print(topic_info.head(20).to_string())

    # Analyze topics over time
    print(f"\nAnalyzing topics over time ({args.nr_bins} bins)...")
    topics_over_time = analyze_topics_over_time(topic_model, df, nr_bins=args.nr_bins)

    # Generate visualization with proper percentages and smoothing
    print(f"\nGenerating visualization...")

    # Calculate actual percentages per time bin
    topics_over_time_pct = topics_over_time.copy()
    total_per_bin = topics_over_time.groupby("Timestamp")["Frequency"].transform("sum")
    topics_over_time_pct["Frequency"] = (topics_over_time["Frequency"] / total_per_bin * 100)

    # Apply smoothing (rolling average) per topic to remove sawtooth pattern
    smoothed_data = []
    for topic_id in topics_over_time_pct["Topic"].unique():
        topic_data = topics_over_time_pct[topics_over_time_pct["Topic"] == topic_id].copy()
        topic_data = topic_data.sort_values("Timestamp")
        # Rolling average to smooth the sawtooth pattern
        topic_data["Frequency"] = topic_data["Frequency"].rolling(window=args.smooth, min_periods=1, center=True).mean().round(2)
        smoothed_data.append(topic_data)

    topics_over_time_smooth = pd.concat(smoothed_data, ignore_index=True)

    # Generate topic labels
    if args.gpt_names:
        print("\nGenerating topic names with GPT...")
        topic_labels = generate_topic_names_gpt(topic_model, df)
    else:
        topic_labels = filter_topic_words(topic_model)
    topic_model.set_topic_labels(topic_labels)

    fig = topic_model.visualize_topics_over_time(
        topics_over_time_smooth,
        top_n_topics=30,
        normalize_frequency=False,  # Already normalized to %
        custom_labels=True,  # Use our filtered labels
    )
    fig.update_layout(yaxis_title="% of reviews")
    fig.write_html(args.output)
    print(f"Saved to {args.output}")

    # Also save topic hierarchy
    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_hierarchy.write_html("topic_hierarchy.html")
    print("Saved topic_hierarchy.html")

    # Save barchart of top words per topic
    fig_barchart = topic_model.visualize_barchart(top_n_topics=30)
    fig_barchart.write_html("topic_words.html")
    print("Saved topic_words.html")


if __name__ == "__main__":
    main()
