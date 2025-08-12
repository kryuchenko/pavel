# 🏗️ PAVEL Proposed Project Structure

## Current Issues
- Mixed logic: core inside `pavel/` but main modules at top level  
- Empty placeholder directories taking space
- Unclear separation between core library and applications
- `cluster/` and `anomaly_detection/` overlap in functionality

## Proposed Structure

```
pavel/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .env.example
├── 📄 .gitignore
├── 📄 pyproject.toml          # Modern Python packaging
│
├── 📁 src/pavel/              # Main Python package (src layout)
│   ├── __init__.py
│   ├── core/                  # Core utilities & config
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── app_config.py
│   │
│   ├── ingestion/             # Stage 2: Data ingestion
│   │   ├── __init__.py
│   │   ├── google_play.py
│   │   ├── rate_limiter.py
│   │   ├── batch_processor.py
│   │   └── scheduler.py
│   │
│   ├── preprocessing/         # Stage 3: Text preprocessing  
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── normalizer.py
│   │   ├── language_detector.py
│   │   ├── sentence_splitter.py
│   │   └── deduplicator.py
│   │
│   ├── classification/        # Stage 4: Complaint classification
│   │   ├── __init__.py
│   │   ├── complaint_classifier.py
│   │   ├── dataset_builder.py
│   │   └── models/
│   │
│   ├── embeddings/            # Stage 5: Vector embeddings
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── pipeline.py
│   │   ├── vector_store.py
│   │   └── semantic_search.py
│   │
│   ├── clustering/            # Stage 7: Bug clustering & anomaly detection
│   │   ├── __init__.py
│   │   ├── cluster_engine.py
│   │   ├── anomaly_detector.py
│   │   ├── trend_analyzer.py
│   │   └── smart_detection.py
│   │
│   ├── search/               # Stage 10: Search API
│   │   ├── __init__.py
│   │   ├── vector_search.py
│   │   └── query_engine.py
│   │
│   └── reporting/            # Stage 11-12: Reports & alerts
│       ├── __init__.py
│       ├── bug_reporter.py
│       └── alerting.py
│
├── 📁 applications/          # Standalone applications
│   ├── cli/                  # Command-line interface
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── commands/
│   │
│   ├── api/                  # REST API server
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── routes/
│   │   └── models/
│   │
│   └── dashboard/            # Web UI (Stage 13)
│       ├── static/
│       ├── templates/
│       └── app.py
│
├── 📁 tests/                 # All tests organized by component
│   ├── unit/                 # Unit tests per module
│   │   ├── test_core/
│   │   ├── test_ingestion/
│   │   ├── test_preprocessing/
│   │   ├── test_embeddings/
│   │   └── test_clustering/
│   │
│   ├── integration/          # Integration tests
│   │   ├── test_pipeline_e2e.py
│   │   ├── test_real_data.py
│   │   └── test_performance.py
│   │
│   └── fixtures/             # Test data & fixtures
│       ├── sample_reviews.json
│       └── test_configs.py
│
├── 📁 deployment/            # Deployment & infrastructure
│   ├── docker/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── docker-compose.prod.yml
│   │
│   ├── kubernetes/           # K8s manifests (future)
│   └── scripts/              # Deployment scripts
│
├── 📁 docs/                  # Documentation
│   ├── architecture.md
│   ├── api_reference.md
│   ├── deployment_guide.md
│   ├── schemas/
│   │   ├── mongodb_schema.js
│   │   └── api_schemas.json
│   └── examples/
│
├── 📁 data/                  # Data & models (gitignored)
│   ├── models/               # Trained models
│   ├── datasets/             # Training datasets  
│   └── cache/                # Temporary cache
│
└── 📁 tools/                 # Development & analysis tools
    ├── dataset_builder.py    # Build complaint classification dataset
    ├── model_trainer.py      # Train models
    ├── data_explorer.py      # Analyze Google Play data
    └── benchmark.py          # Performance benchmarks
```

## Benefits of New Structure

### 1. **Clear Separation of Concerns**
- `src/pavel/` - Core library (installable package)
- `applications/` - Standalone apps using the library
- `tests/` - Comprehensive test organization
- `deployment/` - Infrastructure as code
- `tools/` - Development utilities

### 2. **Modern Python Standards**
- `src/` layout (best practice for Python packages)
- `pyproject.toml` for modern packaging
- Clear package hierarchy with proper `__init__.py`

### 3. **Scalable Organization**
- Unit tests organized by module
- Integration tests separate from unit tests  
- Applications decoupled from core library
- Easy to add new stages/components

### 4. **Professional Structure**
- Follows industry standards for ML/data projects
- Easy onboarding for new developers
- Clear deployment story
- Proper documentation organization

## Migration Strategy

1. **Phase 1**: Reorganize core library into `src/pavel/`
2. **Phase 2**: Consolidate clustering/anomaly detection  
3. **Phase 3**: Reorganize tests by component
4. **Phase 4**: Move applications to separate directory
5. **Phase 5**: Add modern packaging with pyproject.toml

This structure will make PAVEL more maintainable, testable, and deployable.