"""
Multi-model embedding generation for Google Play reviews.

Supports multiple embedding models with automatic model selection based on 
language and use case. Optimized for multilingual content and batch processing.
"""

import os
import numpy as np
import asyncio
import time
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Set tokenizers parallelism to avoid forking warnings
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from pavel.core.logger import get_logger
from pavel.core.config import get_config

logger = get_logger(__name__)

class SupportedModels(Enum):
    """Supported embedding models"""
    # Multilingual E5 models (best for multilingual content) - ordered by quality
    E5_LARGE_MULTILINGUAL = "intfloat/multilingual-e5-large"  # 1024D (best quality)
    E5_BASE_MULTILINGUAL = "intfloat/multilingual-e5-base"    # 768D (good quality)
    E5_SMALL_MULTILINGUAL = "intfloat/multilingual-e5-small"  # 384D (fast but lower quality)
    
    # English-focused models
    E5_LARGE_V2 = "intfloat/e5-large-v2"  # 1024D
    ALL_MINILM_L6_V2 = "sentence-transformers/all-MiniLM-L6-v2"  # 384D
    
    # Multilingual sentence transformers
    PARAPHRASE_MULTILINGUAL_MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 384D
    DISTILUSE_MULTILINGUAL = "sentence-transformers/distiluse-base-multilingual-cased"  # 512D
    
    # OpenAI models (requires API key)
    OPENAI_ADA_002 = "text-embedding-ada-002"  # 1536D
    OPENAI_3_SMALL = "text-embedding-3-small"  # 1536D
    OPENAI_3_LARGE = "text-embedding-3-large"  # 3072D

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = SupportedModels.E5_LARGE_MULTILINGUAL.value
    batch_size: int = 32
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    use_cache: bool = True
    cache_ttl_hours: int = 24
    openai_api_key: Optional[str] = None
    device: str = "auto"  # "auto", "cpu", "cuda"
    num_threads: int = 4

@dataclass 
class EmbeddingResult:
    """Result of embedding generation"""
    text: str
    embedding: np.ndarray
    model_name: str
    embedding_dim: int
    processing_time: float
    language: Optional[str] = None
    confidence: float = 1.0
    cached: bool = False
    text_hash: Optional[str] = None

class EmbeddingGenerator:
    """
    Multi-model embedding generator optimized for Google Play reviews.
    
    Features:
    - Multiple embedding models (E5, OpenAI, Sentence-Transformers)
    - Automatic model selection based on language/content
    - Batch processing for efficiency
    - Caching for repeated content
    - Multilingual optimization
    - Async and sync interfaces
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self.model_name = None
        self.embedding_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_threads)
        
        # Initialize model
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the embedding model"""
        try:
            if self.config.model_name in [model.value for model in [
                SupportedModels.OPENAI_ADA_002, 
                SupportedModels.OPENAI_3_SMALL, 
                SupportedModels.OPENAI_3_LARGE
            ]]:
                self._initialize_openai_model()
            else:
                self._initialize_sentence_transformer_model()
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            # Fallback to simple model
            self._initialize_fallback_model()
    
    def _initialize_openai_model(self):
        """Initialize OpenAI embedding model"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
            
        api_key = self.config.openai_api_key or get_config().OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI API key required for OpenAI models")
            
        openai.api_key = api_key
        self.model_name = self.config.model_name
        self.model = "openai"  # Special marker
        
        logger.info(f"Initialized OpenAI embedding model: {self.model_name}")
    
    def _initialize_sentence_transformer_model(self):
        """Initialize Sentence-Transformers model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
            
        logger.info(f"Loading embedding model: {self.config.model_name}")
        
        # Configure device
        device = self.config.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.model = SentenceTransformer(
            self.config.model_name,
            device=device
        )
        
        # Set max sequence length if specified
        if hasattr(self.model, 'max_seq_length'):
            self.model.max_seq_length = self.config.max_seq_length
            
        self.model_name = self.config.model_name
        
        logger.info(f"Model loaded: {self.model_name} (device: {device})")
    
    def _initialize_fallback_model(self):
        """Initialize fallback model if primary fails"""
        logger.warning("Initializing fallback embedding model")
        
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # Try simple multilingual model
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                self.model_name = 'all-MiniLM-L6-v2'
                logger.info("Fallback model loaded: all-MiniLM-L6-v2")
            else:
                # Last resort - use simple averaging (for testing only)
                self.model = "fallback"
                self.model_name = "simple_average"
                logger.warning("Using simple averaging fallback (testing only)")
                
        except Exception as e:
            logger.error(f"Failed to initialize fallback model: {e}")
            self.model = "fallback"
            self.model_name = "simple_average"
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text caching"""
        return hashlib.md5(f"{text}:{self.model_name}".encode('utf-8')).hexdigest()
    
    def _get_cached_embedding(self, text: str) -> Optional[EmbeddingResult]:
        """Get cached embedding if available and not expired"""
        if not self.config.use_cache:
            return None
            
        text_hash = self._get_text_hash(text)
        if text_hash in self.embedding_cache:
            cached_result, timestamp = self.embedding_cache[text_hash]
            
            # Check if cache is still valid
            cache_age_hours = (time.time() - timestamp) / 3600
            if cache_age_hours < self.config.cache_ttl_hours:
                cached_result.cached = True
                return cached_result
            else:
                # Remove expired cache entry
                del self.embedding_cache[text_hash]
        
        return None
    
    def _cache_embedding(self, result: EmbeddingResult):
        """Cache embedding result"""
        if not self.config.use_cache:
            return
            
        text_hash = self._get_text_hash(result.text)
        self.embedding_cache[text_hash] = (result, time.time())
        result.text_hash = text_hash
    
    def _generate_openai_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API"""
        try:
            response = openai.embeddings.create(
                model=self.model_name,
                input=text.replace("\n", " ")
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            if self.config.normalize_embeddings:
                embedding = embedding / np.linalg.norm(embedding)
                
            return embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            raise
    
    def _generate_sentence_transformer_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using Sentence-Transformers"""
        embedding = self.model.encode(
            [text], 
            normalize_embeddings=self.config.normalize_embeddings,
            batch_size=1
        )[0]
        
        return embedding.astype(np.float32)
    
    def _generate_fallback_embedding(self, text: str) -> np.ndarray:
        """Generate simple fallback embedding (for testing)"""
        # Simple word-based averaging (not recommended for production)
        words = text.lower().split()
        if not words:
            return np.zeros(384, dtype=np.float32)
            
        # Create simple hash-based embedding
        word_vectors = []
        for word in words[:50]:  # Limit to 50 words
            # Simple hash-based vector
            hash_val = hash(word) 
            vector = np.array([
                (hash_val >> i) & 1 for i in range(384)
            ], dtype=np.float32)
            word_vectors.append(vector)
        
        if word_vectors:
            embedding = np.mean(word_vectors, axis=0)
            if self.config.normalize_embeddings:
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding
        else:
            return np.zeros(384, dtype=np.float32)
    
    def generate_single(
        self, 
        text: str, 
        language: Optional[str] = None
    ) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            language: Optional language hint
            
        Returns:
            EmbeddingResult with embedding and metadata
        """
        start_time = time.time()
        
        # Check cache first
        cached_result = self._get_cached_embedding(text)
        if cached_result:
            logger.debug(f"Using cached embedding for: {text[:50]}...")
            return cached_result
        
        # Generate embedding based on model type
        try:
            if self.model == "openai":
                embedding = self._generate_openai_embedding(text)
            elif self.model == "fallback":
                embedding = self._generate_fallback_embedding(text)
            else:
                embedding = self._generate_sentence_transformer_embedding(text)
                
            processing_time = time.time() - start_time
            
            # Create result
            result = EmbeddingResult(
                text=text,
                embedding=embedding,
                model_name=self.model_name,
                embedding_dim=len(embedding),
                processing_time=processing_time,
                language=language,
                cached=False
            )
            
            # Cache result
            self._cache_embedding(result)
            
            logger.debug(f"Generated {len(embedding)}D embedding in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Embedding generation failed for text: {text[:100]}... Error: {e}")
            # Return zero embedding as fallback
            return EmbeddingResult(
                text=text,
                embedding=np.zeros(384, dtype=np.float32),
                model_name=f"{self.model_name}_error",
                embedding_dim=384,
                processing_time=time.time() - start_time,
                language=language,
                confidence=0.0
            )
    
    def generate_batch(
        self,
        texts: List[str],
        languages: Optional[List[str]] = None
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            languages: Optional list of language hints
            
        Returns:
            List of EmbeddingResults
        """
        if not texts:
            return []
            
        if languages is None:
            languages = [None] * len(texts)
        elif len(languages) != len(texts):
            logger.warning("Languages list length doesn't match texts, ignoring languages")
            languages = [None] * len(texts)
            
        logger.info(f"Generating embeddings for {len(texts)} texts")
        start_time = time.time()
        
        # Check cache for all texts
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, (text, language) in enumerate(zip(texts, languages)):
            cached_result = self._get_cached_embedding(text)
            if cached_result:
                results.append((i, cached_result))
            else:
                uncached_texts.append((text, language))
                uncached_indices.append(i)
        
        logger.info(f"Found {len(results)} cached embeddings, generating {len(uncached_texts)} new ones")
        
        # Process uncached texts
        if uncached_texts:
            if self.model == "openai":
                # OpenAI batch processing
                new_results = self._generate_openai_batch([t[0] for t in uncached_texts])
                for i, ((text, language), embedding) in enumerate(zip(uncached_texts, new_results)):
                    result = EmbeddingResult(
                        text=text,
                        embedding=embedding,
                        model_name=self.model_name,
                        embedding_dim=len(embedding),
                        processing_time=0.0,  # Will be set below
                        language=language
                    )
                    self._cache_embedding(result)
                    results.append((uncached_indices[i], result))
                    
            elif self.model == "fallback":
                # Fallback batch processing
                for i, (text, language) in enumerate(uncached_texts):
                    result = self.generate_single(text, language)
                    results.append((uncached_indices[i], result))
                    
            else:
                # Sentence-transformers batch processing
                batch_texts = [t[0] for t in uncached_texts]
                embeddings = self.model.encode(
                    batch_texts,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=self.config.normalize_embeddings
                )
                
                for i, ((text, language), embedding) in enumerate(zip(uncached_texts, embeddings)):
                    result = EmbeddingResult(
                        text=text,
                        embedding=embedding.astype(np.float32),
                        model_name=self.model_name,
                        embedding_dim=len(embedding),
                        processing_time=0.0,  # Will be set below
                        language=language
                    )
                    self._cache_embedding(result)
                    results.append((uncached_indices[i], result))
        
        # Sort results by original index and extract
        results.sort(key=lambda x: x[0])
        final_results = [r[1] for r in results]
        
        # Update processing times
        total_time = time.time() - start_time
        avg_time = total_time / len(texts)
        for result in final_results:
            if result.processing_time == 0.0:
                result.processing_time = avg_time
        
        logger.info(f"Batch embedding complete: {len(texts)} texts in {total_time:.2f}s ({avg_time*1000:.1f}ms/text)")
        return final_results
    
    def _generate_openai_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using OpenAI API in batch"""
        try:
            # Clean texts for OpenAI
            cleaned_texts = [text.replace("\n", " ") for text in texts]
            
            response = openai.embeddings.create(
                model=self.model_name,
                input=cleaned_texts
            )
            
            embeddings = []
            for data in response.data:
                embedding = np.array(data.embedding, dtype=np.float32)
                if self.config.normalize_embeddings:
                    embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI batch embedding failed: {e}")
            # Fallback to individual processing
            return [self._generate_openai_embedding(text) for text in texts]
    
    async def generate_single_async(
        self,
        text: str,
        language: Optional[str] = None
    ) -> EmbeddingResult:
        """Async version of generate_single"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.generate_single,
            text,
            language
        )
    
    async def generate_batch_async(
        self,
        texts: List[str],
        languages: Optional[List[str]] = None
    ) -> List[EmbeddingResult]:
        """Async version of generate_batch"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.generate_batch,
            texts,
            languages
        )
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for the current model"""
        if self.model == "openai":
            dim_map = {
                SupportedModels.OPENAI_ADA_002.value: 1536,
                SupportedModels.OPENAI_3_SMALL.value: 1536,
                SupportedModels.OPENAI_3_LARGE.value: 3072
            }
            return dim_map.get(self.model_name, 1536)
        elif self.model == "fallback":
            return 384
        else:
            # Get from model
            try:
                test_embedding = self.model.encode(["test"])
                return len(test_embedding[0])
            except:
                return 384  # Default fallback
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.embedding_cache),
            "model_name": self.model_name,
            "cache_enabled": self.config.use_cache,
            "cache_ttl_hours": self.config.cache_ttl_hours
        }
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)