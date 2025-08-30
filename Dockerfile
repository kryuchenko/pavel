# PAVEL - Problem & Anomaly Vector Embedding Locator
# Production-ready Docker image

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy models
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download ru_core_news_sm

# Copy application code
COPY src/ ./src/
COPY *.py ./
COPY *.md ./
COPY *.json ./

# Create necessary directories
RUN mkdir -p logs models data secrets

# Set Python path
ENV PYTHONPATH=/app/src

# Expose ports
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from src.pavel.core.config import get_config; get_config()" || exit 1

# Default command - can be overridden
CMD ["python", "search_reviews.py", "--help"]