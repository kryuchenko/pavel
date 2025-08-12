# PAVEL Analysis Tools

## API Key Setup

### Method 1: .env.local (Recommended)
```bash
# Create local env file (gitignored)
cp .env .env.local

# Edit .env.local and replace YOUR_OPENAI_API_KEY_HERE with actual key
vim .env.local
```

### Method 2: GitHub Secret
```bash
# Use wrapper script that fetches from GitHub
./tools/run_with_gh_secret.sh tools/chatgpt_labeling.py
```

### Method 3: Direct export
```bash
export OPENAI_API_KEY="sk-..."
PYTHONPATH=src python tools/chatgpt_labeling.py
```

## Available Tools

### ChatGPT Labeling
Professional complaint classification using GPT-4o-mini:
```bash
PYTHONPATH=src python tools/chatgpt_labeling.py
```
Cost: ~$0.92 for 1,846 reviews

### Full inDriver Ingestion
Comprehensive review collection across 79 locales:
```bash
PYTHONPATH=src python tools/full_indrive_ingestion.py
```

### Complaint Classifier Training
Train ML model on labeled dataset:
```bash
PYTHONPATH=src python tools/train_complaint_classifier.py
```

### Field Analysis
- `explore_fields.py` - Analyzes Google Play scraper fields
- `field_analysis_results.json` - Field analysis results