"""
Complete embedding pipeline integrating preprocessing and vector generation.

Orchestrates the full flow from raw Google Play reviews to semantic vectors
ready for anomaly detection and similarity analysis.
"""

import asyncio
import time
import uuid
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from pavel.core.logger import get_logger
from pavel.core.config import get_config

# Import preprocessing components
from ..preprocessing import (
    PreprocessingPipeline,
    ProcessedReview,
    ProcessingBatchResult
)

# Import embedding components
from .embedding_generator import EmbeddingGenerator, EmbeddingConfig, EmbeddingResult
from .vector_store import VectorStore, VectorStoreConfig
from .semantic_search import SemanticSearchEngine

logger = get_logger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the complete embedding pipeline"""
    # Embedding configuration
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_batch_size: int = 32
    normalize_embeddings: bool = True
    
    # Vector storage configuration
    vector_collection: str = "review_embeddings"
    similarity_metric: str = "cosine"
    
    # Processing configuration
    max_concurrent_batches: int = 4
    batch_size: int = 100
    enable_preprocessing: bool = True
    
    # Performance options
    use_cache: bool = True
    store_intermediate_results: bool = True
    
    # Quality filters
    min_text_length: int = 10
    max_text_length: int = 2000
    min_language_confidence: float = 0.6

@dataclass
class EmbeddingBatch:
    """Batch of reviews processed through the complete pipeline"""
    batch_id: str
    app_id: str
    reviews: List[Dict[str, Any]]
    processed_reviews: List[ProcessedReview]
    embeddings: List[EmbeddingResult]
    stored_count: int
    processing_time: float
    errors: List[str]

@dataclass
class PipelineResult:
    """Result of complete pipeline processing"""
    app_id: str
    total_reviews: int
    processed_reviews: int
    embedded_reviews: int
    stored_reviews: int
    failed_reviews: int
    processing_time: float
    batches: List[EmbeddingBatch]
    language_distribution: Dict[str, int]
    quality_metrics: Dict[str, Any]

class EmbeddingPipeline:
    """
    Complete embedding pipeline for Google Play reviews.
    
    Integrates:
    1. Text preprocessing (normalization, language detection, segmentation)
    2. Embedding generation (E5, OpenAI, or other models)
    3. Vector storage (MongoDB with similarity search)
    4. Semantic search capabilities
    
    Features:
    - End-to-end processing from raw reviews to searchable vectors
    - Batch processing with concurrent execution
    - Quality filtering and validation
    - Comprehensive error handling and recovery
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize preprocessing pipeline
        if self.config.enable_preprocessing:
            self.preprocessing_pipeline = PreprocessingPipeline()
        else:
            self.preprocessing_pipeline = None
        
        # Initialize embedding components
        embedding_config = EmbeddingConfig(
            model_name=self.config.embedding_model,
            batch_size=self.config.embedding_batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            use_cache=self.config.use_cache
        )
        
        vector_config = VectorStoreConfig(
            collection_name=self.config.vector_collection,
            similarity_metric=self.config.similarity_metric
        )
        
        self.embedding_generator = EmbeddingGenerator(embedding_config)
        self.vector_store = VectorStore(vector_config)
        self.semantic_search = SemanticSearchEngine(
            embedding_generator=self.embedding_generator,
            vector_store=self.vector_store
        )
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_batches)
    
    async def process_app_reviews(self, 
                                 app_id: str,
                                 reviews: List[Dict[str, Any]]) -> PipelineResult:
        """
        Process all reviews for an app through the complete pipeline.
        
        Args:
            app_id: Application identifier
            reviews: List of raw review dictionaries
            
        Returns:
            Complete pipeline processing result
        """
        logger.info(f"Starting pipeline processing for app {app_id}: {len(reviews)} reviews")
        start_time = time.time()
        
        # Split reviews into batches
        batches = self._create_batches(app_id, reviews)
        logger.info(f"Created {len(batches)} batches for processing")
        
        # Process batches concurrently
        processed_batches = []
        
        if self.config.max_concurrent_batches > 1:
            # Concurrent processing
            processed_batches = await self._process_batches_concurrent(batches)
        else:
            # Sequential processing
            processed_batches = await self._process_batches_sequential(batches)
        
        # Compile results
        result = self._compile_pipeline_result(app_id, processed_batches, time.time() - start_time)
        
        logger.info(f"Pipeline processing complete for {app_id}: "
                   f"{result.stored_reviews}/{result.total_reviews} reviews processed in {result.processing_time:.2f}s")
        
        return result
    
    async def process_single_review(self,
                                   review_data: Dict[str, Any]) -> Optional[str]:
        """
        Process a single review through the complete pipeline.
        
        Args:
            review_data: Raw review data
            
        Returns:
            Vector ID if successful, None if failed
        """
        try:
            # Step 1: Preprocessing (if enabled)
            if self.preprocessing_pipeline:
                processed_review = await self.preprocessing_pipeline.process_single_review(review_data)
                text = processed_review.normalized_content
                language = processed_review.detected_language
                metadata = {
                    'app_id': processed_review.app_id,
                    'review_id': processed_review.review_id,
                    'language': language,
                    'sentence_count': processed_review.sentence_count,
                    'processed_at': processed_review.processed_at
                }
            else:
                # Direct processing without preprocessing
                text = review_data.get('content', '')
                language = None
                metadata = {
                    'app_id': review_data.get('app_id'),
                    'review_id': review_data.get('review_id'),
                    'raw_processing': True
                }
            
            # Quality checks
            if not self._passes_quality_filters(text, language):
                logger.debug(f"Review {review_data.get('review_id')} failed quality filters")
                return None
            
            # Step 2: Generate embedding
            embedding_result = await self.embedding_generator.generate_single_async(text, language)
            
            # Step 3: Store vector
            vector_id = f"{metadata['app_id']}_{metadata['review_id']}"
            success = await self.vector_store.store_embedding_async(
                vector_id, 
                embedding_result,
                metadata
            )
            
            if success:
                logger.debug(f"Successfully processed review: {vector_id}")
                return vector_id
            else:
                logger.debug(f"Failed to store review: {vector_id}")
                return None
                
        except Exception as e:
            logger.error(f"Single review processing failed: {e}")
            return None
    
    def _create_batches(self, app_id: str, reviews: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Split reviews into processing batches"""
        batches = []
        for i in range(0, len(reviews), self.config.batch_size):
            batch = reviews[i:i + self.config.batch_size]
            batches.append(batch)
        return batches
    
    async def _process_batches_concurrent(self, batches: List[List[Dict[str, Any]]]) -> List[EmbeddingBatch]:
        """Process batches concurrently"""
        tasks = []
        for i, batch in enumerate(batches):
            task = self._process_single_batch(f"batch_{i}", batch[0].get('app_id'), batch)
            tasks.append(task)
        
        # Execute batches concurrently but limit concurrency
        results = []
        for i in range(0, len(tasks), self.config.max_concurrent_batches):
            batch_tasks = tasks[i:i + self.config.max_concurrent_batches]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing failed: {result}")
                else:
                    results.append(result)
        
        return results
    
    async def _process_batches_sequential(self, batches: List[List[Dict[str, Any]]]) -> List[EmbeddingBatch]:
        """Process batches sequentially"""
        results = []
        for i, batch in enumerate(batches):
            try:
                result = await self._process_single_batch(f"batch_{i}", batch[0].get('app_id'), batch)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch {i} processing failed: {e}")
        
        return results
    
    async def _process_single_batch(self, 
                                   batch_id: str,
                                   app_id: str,
                                   reviews: List[Dict[str, Any]]) -> EmbeddingBatch:
        """Process a single batch through the complete pipeline"""
        start_time = time.time()
        errors = []
        
        logger.debug(f"Processing batch {batch_id}: {len(reviews)} reviews")
        
        # Step 1: Preprocessing (if enabled)
        processed_reviews = []
        if self.preprocessing_pipeline:
            try:
                for review in reviews:
                    processed_review = await self.preprocessing_pipeline.process_single_review(review)
                    processed_reviews.append(processed_review)
            except Exception as e:
                errors.append(f"Preprocessing failed: {e}")
                logger.error(f"Batch {batch_id} preprocessing failed: {e}")
                return EmbeddingBatch(
                    batch_id=batch_id,
                    app_id=app_id,
                    reviews=reviews,
                    processed_reviews=[],
                    embeddings=[],
                    stored_count=0,
                    processing_time=time.time() - start_time,
                    errors=errors
                )
        else:
            # Create minimal processed reviews
            for review in reviews:
                processed_reviews.append(type('ProcessedReview', (), {
                    'normalized_content': review.get('content', ''),
                    'detected_language': None,
                    'review_id': review.get('review_id'),
                    'app_id': review.get('app_id')
                })())
        
        # Step 2: Quality filtering
        valid_reviews = []
        for processed_review in processed_reviews:
            if self._passes_quality_filters(
                processed_review.normalized_content,
                getattr(processed_review, 'detected_language', None)
            ):
                valid_reviews.append(processed_review)
        
        logger.debug(f"Batch {batch_id}: {len(valid_reviews)}/{len(processed_reviews)} reviews passed quality filters")
        
        # Step 3: Generate embeddings
        embeddings = []
        if valid_reviews:
            try:
                texts = [r.normalized_content for r in valid_reviews]
                languages = [getattr(r, 'detected_language', None) for r in valid_reviews]
                
                embedding_results = await self.embedding_generator.generate_batch_async(texts, languages)
                embeddings = embedding_results
                
            except Exception as e:
                errors.append(f"Embedding generation failed: {e}")
                logger.error(f"Batch {batch_id} embedding generation failed: {e}")
        
        # Step 4: Store embeddings
        stored_count = 0
        if embeddings and valid_reviews:
            try:
                # Prepare batch for storage with unique IDs
                storage_batch = []
                for i, (processed_review, embedding) in enumerate(zip(valid_reviews, embeddings)):
                    # Create unique vector ID with timestamp to avoid duplicates
                    timestamp = int(time.time() * 1000000)  # microsecond precision
                    vector_id = f"{processed_review.app_id}_{processed_review.review_id}_{timestamp}_{i}"
                    
                    metadata = {
                        'app_id': processed_review.app_id,
                        'review_id': processed_review.review_id,
                        'language': getattr(processed_review, 'detected_language', None),
                        'batch_id': batch_id,
                        'unique_id': vector_id
                    }
                    storage_batch.append((vector_id, embedding, metadata))
                
                # Store batch
                storage_stats = self.vector_store.store_embeddings_batch(storage_batch)
                stored_count = storage_stats['stored']
                
                if storage_stats['errors'] > 0:
                    errors.append(f"Storage errors: {storage_stats['errors']}")
                
            except Exception as e:
                errors.append(f"Storage failed: {e}")
                logger.error(f"Batch {batch_id} storage failed: {e}")
        
        # Create batch result
        batch_result = EmbeddingBatch(
            batch_id=batch_id,
            app_id=app_id,
            reviews=reviews,
            processed_reviews=processed_reviews,
            embeddings=embeddings,
            stored_count=stored_count,
            processing_time=time.time() - start_time,
            errors=errors
        )
        
        logger.debug(f"Batch {batch_id} complete: {stored_count} embeddings stored in {batch_result.processing_time:.2f}s")
        return batch_result
    
    def _passes_quality_filters(self, text: str, language: Optional[str]) -> bool:
        """Check if text passes quality filters"""
        if not text or len(text.strip()) < self.config.min_text_length:
            return False
        
        if len(text) > self.config.max_text_length:
            return False
        
        # Language confidence check (if available)
        # This would require access to language confidence from preprocessing
        # For now, we'll accept all languages
        
        return True
    
    def _compile_pipeline_result(self,
                                app_id: str,
                                batches: List[EmbeddingBatch],
                                total_time: float) -> PipelineResult:
        """Compile final pipeline result from batch results"""
        
        # Aggregate statistics
        total_reviews = sum(len(batch.reviews) for batch in batches)
        processed_reviews = sum(len(batch.processed_reviews) for batch in batches)
        embedded_reviews = sum(len(batch.embeddings) for batch in batches)
        stored_reviews = sum(batch.stored_count for batch in batches)
        failed_reviews = total_reviews - stored_reviews
        
        # Language distribution
        language_dist = {}
        for batch in batches:
            for processed_review in batch.processed_reviews:
                lang = getattr(processed_review, 'detected_language', 'unknown') or 'unknown'
                language_dist[lang] = language_dist.get(lang, 0) + 1
        
        # Quality metrics
        quality_metrics = {
            'preprocessing_success_rate': (processed_reviews / total_reviews) if total_reviews > 0 else 0,
            'embedding_success_rate': (embedded_reviews / processed_reviews) if processed_reviews > 0 else 0,
            'storage_success_rate': (stored_reviews / embedded_reviews) if embedded_reviews > 0 else 0,
            'overall_success_rate': (stored_reviews / total_reviews) if total_reviews > 0 else 0,
            'average_batch_time': total_time / len(batches) if batches else 0,
            'reviews_per_second': stored_reviews / total_time if total_time > 0 else 0
        }
        
        return PipelineResult(
            app_id=app_id,
            total_reviews=total_reviews,
            processed_reviews=processed_reviews,
            embedded_reviews=embedded_reviews,
            stored_reviews=stored_reviews,
            failed_reviews=failed_reviews,
            processing_time=total_time,
            batches=batches,
            language_distribution=language_dist,
            quality_metrics=quality_metrics
        )
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        return {
            'embedding_generator': self.embedding_generator.get_cache_stats(),
            'vector_store': self.vector_store.get_statistics(),
            'semantic_search': self.semantic_search.get_search_statistics(),
            'configuration': {
                'embedding_model': self.config.embedding_model,
                'batch_size': self.config.batch_size,
                'max_concurrent_batches': self.config.max_concurrent_batches,
                'preprocessing_enabled': self.config.enable_preprocessing
            }
        }
    
    def clear_caches(self):
        """Clear all caches"""
        self.embedding_generator.clear_cache()
        self.semantic_search.clear_caches()
        logger.info("Pipeline caches cleared")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)