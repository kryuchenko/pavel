# PAVEL Installation Guide

## Quick Setup

### 1. Install Python Dependencies
```bash
cd /Users/andrey/Projects/pavel
pip install -r requirements.txt
```

### 2. Setup MongoDB
```bash
# Via Homebrew (macOS)
brew tap mongodb/brew
brew install mongodb-community@8.0
brew services start mongodb/brew/mongodb-community@8.0

# Or via Docker
docker-compose up -d mongodb
```

### 3. Verify Installation
```bash
python test_requirements.py
```

### 4. Test Review Collection
```bash
python temp_ingest_script.py
```

## Advanced Setup

### GPU Support (Optional)
For faster ML processing:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Language Models (Optional)
For advanced text processing:
```bash
python -m spacy download en_core_web_sm
python -m spacy download ru_core_news_sm
```

### Environment Variables
Copy `.env` example and configure:
```bash
# MongoDB is already configured for localhost
# Add OpenAI key if using GPT features:
# OPENAI_API_KEY=your_api_key_here
```

## What Gets Installed

- **Google Play Scraper** (1.2.7) - Latest version
- **MongoDB** support with all features  
- **Machine Learning** stack (scikit-learn, torch, transformers)
- **Text Processing** (sentence-transformers, langdetect)
- **Advanced Clustering** (HDBSCAN)
- **Development Tools** (pytest, jupyter)

## Troubleshooting

If installation fails:
1. Update pip: `pip install --upgrade pip`
2. Install build tools: `pip install build wheel`
3. Install dependencies one by one to identify issues

## All Done! 🎉

Your PAVEL installation is ready to collect and analyze app reviews from Google Play Store.