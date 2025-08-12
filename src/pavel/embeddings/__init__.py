"""
PAVEL Stage 4: Vector Embeddings

Semantic vector representation system for Google Play reviews.
Provides high-quality embeddings for anomaly detection and similarity search.
"""

from .embedding_generator import (
    EmbeddingGenerator,
    EmbeddingResult,
    EmbeddingConfig,
    SupportedModels
)

from .vector_store import (
    VectorStore,
    VectorSearchResult,
    VectorStoreConfig,
    SearchQuery
)

from .semantic_search import (
    SemanticSearchEngine,
    SearchResult,
    SimilarityMetric
)

from .embedding_pipeline import (
    EmbeddingPipeline,
    PipelineConfig,
    EmbeddingBatch
)

__all__ = [
    # Core embedding generation
    'EmbeddingGenerator',
    'EmbeddingResult', 
    'EmbeddingConfig',
    'SupportedModels',
    
    # Vector storage and retrieval
    'VectorStore',
    'VectorSearchResult',
    'VectorStoreConfig',
    'SearchQuery',
    
    # Semantic search
    'SemanticSearchEngine',
    'SearchResult',
    'SimilarityMetric',
    
    # Complete pipeline
    'EmbeddingPipeline',
    'PipelineConfig',
    'EmbeddingBatch'
]

# Module metadata
__version__ = "1.0.0"
__description__ = "PAVEL Stage 4: Vector Embeddings for Google Play Reviews"