.PHONY: setup run stop clean dev logs migrate download-model check-deps help ingest embed install-llama llama-start llama-stop

# Default target
help:
	@echo "PAVEL - Problem & Anomaly Vector Embedding Locator"
	@echo ""
	@echo "Usage:"
	@echo "  make setup    - First time setup (download model, start DB, build)"
	@echo "  make run      - Start all services"
	@echo "  make stop     - Stop all services"
	@echo "  make dev      - Run in development mode"
	@echo "  make logs     - Show logs"
	@echo "  make clean    - Remove all data and containers"
	@echo ""

# === SETUP ===

setup: check-deps install-llama download-model docker-up migrate build
	@echo ""
	@echo "✓ PAVEL is ready!"
	@echo ""
	@echo "  Run 'make run' to start"
	@echo "  API: http://localhost:8080"
	@echo "  DB:  localhost:5432"
	@echo ""

check-deps:
	@echo "Checking dependencies..."
	@command -v go >/dev/null 2>&1 || { echo "ERROR: Go is not installed. Install: brew install go"; exit 1; }
	@command -v docker >/dev/null 2>&1 || { echo "ERROR: Docker is not installed. Install: brew install docker"; exit 1; }
	@echo "✓ All dependencies found"

ensure-docker:
	@if ! docker info >/dev/null 2>&1; then \
		echo "Starting Docker..."; \
		open -a Docker; \
		echo "Waiting for Docker to start..."; \
		while ! docker info >/dev/null 2>&1; do sleep 1; done; \
		echo "✓ Docker is running"; \
	else \
		echo "✓ Docker is running"; \
	fi

download-model:
	@echo "Checking model..."
	@mkdir -p models
	@if [ ! -f models/embeddinggemma-300M-Q8_0.gguf ]; then \
		echo "Downloading EmbeddingGemma 300M (~313MB)..."; \
		curl -L --progress-bar -o models/embeddinggemma-300M-Q8_0.gguf \
			"https://huggingface.co/unsloth/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf"; \
		echo "✓ Model downloaded"; \
	else \
		echo "✓ Model already exists"; \
	fi

# === DOCKER ===

docker-up: ensure-docker
	@echo "Starting PostgreSQL..."
	@docker-compose up -d db
	@echo "Waiting for PostgreSQL to be ready..."
	@sleep 3
	@until docker-compose exec -T db pg_isready -U pavel > /dev/null 2>&1; do \
		sleep 1; \
	done
	@echo "✓ PostgreSQL is ready"

migrate:
	@echo "Running migrations..."
	@docker-compose exec -T db psql -U pavel -d pavel -f /docker-entrypoint-initdb.d/001_init.sql 2>/dev/null || true
	@echo "✓ Migrations complete"

# === LLAMA-SERVER ===

install-llama:
	@echo "Checking llama.cpp..."
	@if ! command -v llama-server >/dev/null 2>&1; then \
		echo "Installing llama.cpp..."; \
		brew install llama.cpp; \
	else \
		echo "✓ llama.cpp already installed"; \
	fi

llama-start: install-llama
	@echo "Starting llama-server..."
	@if ! pgrep -f "llama-server.*8090" > /dev/null; then \
		llama-server -m models/embeddinggemma-300M-Q8_0.gguf --embeddings --port 8090 > /dev/null 2>&1 & \
		sleep 2; \
		echo "✓ llama-server started on :8090"; \
	else \
		echo "✓ llama-server already running"; \
	fi

llama-stop:
	@pkill -f "llama-server.*8090" 2>/dev/null || true
	@echo "✓ llama-server stopped"

# === BUILD ===

build:
	@echo "Building Go binaries..."
	@go build -o bin/pavel ./cmd/pavel
	@echo "✓ Build complete"

# === RUN ===

run:
	@docker-compose up -d
	@echo "PAVEL is running"
	@echo "  API: http://localhost:8080"

ingest: docker-up
	@./bin/pavel ingest

embed: docker-up llama-start
	@./bin/pavel embed

dev:
	@docker-compose up -d db
	@go run ./cmd/pavel

stop:
	@docker-compose down
	@echo "PAVEL stopped"

logs:
	@docker-compose logs -f

# === CLEAN ===

clean:
	@echo "Cleaning up..."
	@docker-compose down -v 2>/dev/null || true
	@rm -rf bin/
	@echo "✓ Cleaned (models preserved)"

clean-all: clean
	@rm -rf models/
	@echo "✓ Models removed"

# === PYTHON ANALYTICS ===

analytics-setup:
	@echo "Setting up Python analytics..."
	@pip install -r scripts/requirements.txt
	@echo "✓ Python dependencies installed"

visualize:
	@python scripts/export_embeddings.py
	@embedding-atlas data/embeddings.parquet

trends:
	@python scripts/analyze_trends.py
