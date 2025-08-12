# PAVEL Tests

Comprehensive test suite for all PAVEL pipeline components and stages.

## Test Organization

### Stage Tests (Validate individual pipeline stages)
```
tests/
├── test_stage0.py              # Stage 0: Environment setup & access
├── test_stage1.py              # Stage 1: MongoDB schema & indexing  
├── test_stage2.py              # Stage 2: Google Play ingestion
├── test_stage2_simple.py       # Stage 2: Core ingestion (no external API)
├── test_stage3.py              # Stage 3: Text preprocessing & normalization
├── test_stage4.py              # Stage 4: Complaint detection (TODO)
├── test_stage4_final.py        # Stage 4: Final validation (TODO)
└── test_stage5_*.py            # Stage 5+: Smart anomaly detection suite
```

### Stage 5+ Smart Detection Tests
```
├── test_stage5_smart_detection.py  # Core smart detection with synthetic data
├── test_stage5_clusters.py         # Cluster formation validation  
├── test_stage5_large_scale.py      # Production-scale performance (1600+ reviews)
└── test_stage5_complete.py         # Complete pipeline integration
```

### Integration & Performance Tests
```
├── test_integration_main.py        # Main system integration test
├── test_integration_real_data.py   # Real Google Play data integration
├── test_similarity_tuning.py       # Embedding similarity tuning
├── test_embedding_compatibility.py # Embedding model compatibility
└── test_multilingual_extended.py   # Extended multilingual testing
```

### Language & Specialized Tests  
```
├── test_language_detection_real.py     # Real language detection accuracy
└── test_kazakh_detection_alternatives.py # Kazakh language alternatives
```

## Usage

### Run Stage-Specific Tests
```bash
# Test individual stages
python tests/test_stage0.py
python tests/test_stage1.py  
python tests/test_stage2.py
python tests/test_stage3.py

# Test Stage 5+ smart detection
python tests/test_stage5_smart_detection.py
python tests/test_stage5_clusters.py
python tests/test_stage5_large_scale.py  # Takes ~2-3 minutes
```

### Run Integration Tests
```bash
python tests/test_integration_main.py        # Quick system validation
python tests/test_integration_real_data.py   # With real Google Play data
```

### Run All Tests (pytest)
```bash
# All tests
pytest tests/

# Specific test patterns  
pytest tests/test_stage5_*                   # All Stage 5 tests
pytest tests/test_integration_*              # Integration tests only
pytest -v tests/test_stage5_large_scale.py   # Verbose large-scale test
```

## Test Categories & Status

### ✅ **Working Tests (Production Ready)**
- **Stage 0-3**: Environment, MongoDB, Ingestion, Preprocessing
- **Stage 5+**: Smart anomaly detection with real data validation
- **Integration**: Main system integration and real data processing
- **Performance**: Large-scale testing (1,600+ reviews across 8 languages)

### 🚧 **Tests in Development**  
- **Stage 4**: Complaint classification (waiting for implementation)
- **Stage 7-14**: Advanced features (clustering, search API, dashboard)

### 📊 **Test Results Summary**
- **Total Tests**: 18 test files
- **Core Pipeline**: All stages 0,1,2,3,5,6 passing
- **Smart Detection**: Full validation with synthetic and real data
- **Performance**: Production-scale metrics verified
- **Languages**: 8 languages tested (EN,RU,ES,PT,FR,DE,IT,TR)

## Performance Benchmarks

### Large Scale Test Results
```
📊 Tested with inDrive (sinet.startup.inDriver):
- 1,600+ real reviews across 8 languages
- 50+ reviews/second end-to-end processing  
- 118+ embeddings/second generation
- 5 semantic clusters formed automatically
- Health scoring and anomaly detection working
```

## Test Dependencies
- MongoDB running on localhost:27017 (use Docker)
- Google Play API access (no auth required for public reviews)
- Internet connection for API calls
- Python dependencies: `pip install -r requirements.txt`