"""
MongoDB-based vector storage for Google Play review embeddings.

Uses MongoDB 8+ native capabilities for vector storage and similarity search.
Supports both Atlas Vector Search and standard MongoDB operations.
"""

import numpy as np
import asyncio
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError, BulkWriteError

from pavel.core.logger import get_logger
from pavel.core.config import get_config
from .embedding_generator import EmbeddingResult

logger = get_logger(__name__)

class SimilarityMetric(Enum):
    """Supported similarity metrics"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean" 
    DOT_PRODUCT = "dotProduct"
    MANHATTAN = "manhattan"

@dataclass
class VectorStoreConfig:
    """Configuration for vector storage"""
    collection_name: str = "embeddings"
    index_name: str = "vector_index"
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
    dimension: int = 1024
    use_atlas_vector_search: bool = False
    create_indexes: bool = True
    batch_size: int = 100

@dataclass
class SearchQuery:
    """Vector search query"""
    vector: np.ndarray
    limit: int = 10
    min_similarity: float = 0.0
    filters: Optional[Dict[str, Any]] = None
    include_metadata: bool = True

@dataclass
class VectorSearchResult:
    """Result from vector search"""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

@dataclass
class VectorDocument:
    """Document structure for vector storage"""
    _id: str
    text: str
    embedding: List[float]
    embedding_model: str
    embedding_dim: int
    language: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime
    text_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        doc = asdict(self)
        # Ensure datetime is timezone-aware
        if not doc['created_at'].tzinfo:
            doc['created_at'] = doc['created_at'].replace(tzinfo=timezone.utc)
        return doc

class VectorStore:
    """
    MongoDB-based vector storage for embeddings.
    
    Features:
    - Native MongoDB 8+ vector operations
    - Optional Atlas Vector Search support
    - Efficient similarity search
    - Metadata filtering
    - Batch operations
    - Duplicate detection
    """
    
    def __init__(self, 
                 config: Optional[VectorStoreConfig] = None,
                 mongo_client: Optional[MongoClient] = None):
        self.config = config or VectorStoreConfig()
        self.mongo_config = get_config()
        self.client = mongo_client or self._get_mongo_client()
        self.db = self.client[self.mongo_config.MONGODB_DATABASE]
        self.collection = self.db[self.config.collection_name]
        
        # Initialize indexes
        if self.config.create_indexes:
            self._create_indexes()
    
    def _get_mongo_client(self) -> MongoClient:
        """Get MongoDB client"""
        return MongoClient(
            self.mongo_config.MONGODB_URI,
            serverSelectionTimeoutMS=5000
        )
    
    def _create_indexes(self):
        """Create necessary indexes for vector operations"""
        try:
            # Text hash index for duplicate detection
            self.collection.create_index("text_hash", unique=True, sparse=True)
            
            # Metadata indexes for filtering
            self.collection.create_index("language")
            self.collection.create_index("embedding_model")
            self.collection.create_index("created_at")
            self.collection.create_index("metadata.app_id")
            self.collection.create_index("metadata.review_id")
            
            # Compound indexes for common queries
            self.collection.create_index([
                ("language", 1), 
                ("embedding_model", 1),
                ("created_at", -1)
            ])
            
            logger.info(f"Created indexes for collection: {self.config.collection_name}")
            
            # Try to create vector index if Atlas Vector Search is available
            if self.config.use_atlas_vector_search:
                self._create_atlas_vector_index()
                
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
    
    def _create_atlas_vector_index(self):
        """Create Atlas Vector Search index"""
        try:
            # This would be configured through MongoDB Atlas UI or API
            # Here we just log the intended configuration
            vector_index_spec = {
                "fields": [{
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": self.config.dimension,
                    "similarity": self.config.similarity_metric.value if hasattr(self.config.similarity_metric, 'value') else str(self.config.similarity_metric)
                }, {
                    "type": "filter", 
                    "path": "language"
                }, {
                    "type": "filter",
                    "path": "embedding_model"
                }]
            }
            
            logger.info(f"Atlas Vector Search index spec: {vector_index_spec}")
            logger.info("Note: Atlas Vector Search index must be created through MongoDB Atlas UI")
            
        except Exception as e:
            logger.warning(f"Atlas Vector Search index creation failed: {e}")
    
    def store_embedding(self, 
                       id: str,
                       embedding_result: EmbeddingResult,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a single embedding.
        
        Args:
            id: Unique identifier
            embedding_result: Result from embedding generation
            metadata: Additional metadata
            
        Returns:
            True if stored successfully, False if duplicate
        """
        try:
            doc = VectorDocument(
                _id=id,
                text=embedding_result.text,
                embedding=embedding_result.embedding.tolist(),
                embedding_model=embedding_result.model_name,
                embedding_dim=embedding_result.embedding_dim,
                language=embedding_result.language,
                metadata=metadata or {},
                created_at=datetime.now(timezone.utc),
                text_hash=embedding_result.text_hash
            )
            
            self.collection.insert_one(doc.to_dict())
            logger.debug(f"Stored embedding for ID: {id}")
            return True
            
        except DuplicateKeyError:
            logger.debug(f"Duplicate embedding skipped: {id}")
            return False
        except Exception as e:
            logger.error(f"Failed to store embedding {id}: {e}")
            return False
    
    def store_embeddings_batch(self, 
                              embeddings: List[Tuple[str, EmbeddingResult, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Store multiple embeddings in batch.
        
        Args:
            embeddings: List of (id, embedding_result, metadata) tuples
            
        Returns:
            Statistics about the batch operation
        """
        if not embeddings:
            return {"stored": 0, "duplicates": 0, "errors": 0}
        
        logger.info(f"Storing batch of {len(embeddings)} embeddings")
        
        documents = []
        for id, embedding_result, metadata in embeddings:
            doc = VectorDocument(
                _id=id,
                text=embedding_result.text,
                embedding=embedding_result.embedding.tolist(),
                embedding_model=embedding_result.model_name,
                embedding_dim=embedding_result.embedding_dim,
                language=embedding_result.language,
                metadata=metadata or {},
                created_at=datetime.now(timezone.utc),
                text_hash=embedding_result.text_hash
            )
            documents.append(doc.to_dict())
        
        # Batch insert with duplicate handling
        stored = 0
        duplicates = 0
        errors = 0
        
        try:
            result = self.collection.insert_many(documents, ordered=False)
            stored = len(result.inserted_ids)
            
        except BulkWriteError as e:
            stored = len(e.details.get('writeErrors', []))
            
            # Count duplicates and other errors
            for error in e.details.get('writeErrors', []):
                if error['code'] == 11000:  # Duplicate key error
                    duplicates += 1
                else:
                    errors += 1
                    
            stored = len(embeddings) - duplicates - errors
            
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            errors = len(embeddings)
        
        stats = {
            "stored": stored,
            "duplicates": duplicates, 
            "errors": errors,
            "total": len(embeddings)
        }
        
        logger.info(f"Batch storage complete: {stats}")
        return stats
    
    def search_similar(self, query: SearchQuery) -> List[VectorSearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query: Search query with vector and parameters
            
        Returns:
            List of similar vector results
        """
        if self.config.use_atlas_vector_search:
            return self._search_atlas_vector(query)
        else:
            return self._search_mongodb_native(query)
    
    def _search_atlas_vector(self, query: SearchQuery) -> List[VectorSearchResult]:
        """Search using Atlas Vector Search"""
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.config.index_name,
                        "path": "embedding", 
                        "queryVector": query.vector.tolist(),
                        "numCandidates": query.limit * 10,
                        "limit": query.limit
                    }
                },
                {
                    "$addFields": {
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            # Add filters if specified
            if query.filters:
                pipeline.insert(1, {"$match": query.filters})
            
            # Add minimum similarity filter
            if query.min_similarity > 0:
                pipeline.append({"$match": {"score": {"$gte": query.min_similarity}}})
            
            # Project fields
            project_fields = {
                "_id": 1,
                "text": 1,
                "metadata": 1,
                "score": 1
            }
            
            if query.include_metadata:
                project_fields.update({
                    "language": 1,
                    "embedding_model": 1,
                    "created_at": 1
                })
            
            pipeline.append({"$project": project_fields})
            
            results = []
            for doc in self.collection.aggregate(pipeline):
                result = VectorSearchResult(
                    id=doc["_id"],
                    score=doc["score"],
                    text=doc["text"], 
                    metadata=doc.get("metadata", {})
                )
                
                if query.include_metadata:
                    result.metadata.update({
                        "language": doc.get("language"),
                        "embedding_model": doc.get("embedding_model"),
                        "created_at": doc.get("created_at")
                    })
                
                results.append(result)
            
            logger.debug(f"Atlas Vector Search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Atlas Vector Search failed: {e}")
            # Fallback to native search
            return self._search_mongodb_native(query)
    
    def _search_mongodb_native(self, query: SearchQuery) -> List[VectorSearchResult]:
        """Search using standard MongoDB operations with manual similarity calculation"""
        try:
            # Build filter query
            filter_query = {}
            if query.filters:
                filter_query.update(query.filters)
            
            # Get candidates (potentially with basic filtering)
            projection = {
                "_id": 1,
                "text": 1, 
                "embedding": 1,
                "metadata": 1
            }
            
            if query.include_metadata:
                projection.update({
                    "language": 1,
                    "embedding_model": 1,
                    "created_at": 1
                })
            
            # Limit candidate set for efficiency
            candidate_limit = min(query.limit * 100, 10000)
            cursor = self.collection.find(filter_query, projection).limit(candidate_limit)
            
            # Calculate similarities manually
            candidates = []
            query_vector = query.vector
            
            for doc in cursor:
                doc_vector = np.array(doc["embedding"], dtype=np.float32)
                
                # Calculate similarity based on metric
                if self.config.similarity_metric == SimilarityMetric.COSINE:
                    # Cosine similarity
                    dot_product = np.dot(query_vector, doc_vector)
                    norm_query = np.linalg.norm(query_vector)
                    norm_doc = np.linalg.norm(doc_vector)
                    similarity = dot_product / (norm_query * norm_doc + 1e-8)
                    
                elif self.config.similarity_metric == SimilarityMetric.DOT_PRODUCT:
                    similarity = np.dot(query_vector, doc_vector)
                    
                elif self.config.similarity_metric == SimilarityMetric.EUCLIDEAN:
                    distance = np.linalg.norm(query_vector - doc_vector)
                    similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                    
                elif self.config.similarity_metric == SimilarityMetric.MANHATTAN:
                    distance = np.sum(np.abs(query_vector - doc_vector))
                    similarity = 1.0 / (1.0 + distance)
                    
                else:
                    similarity = np.dot(query_vector, doc_vector)  # Default to dot product
                
                # Filter by minimum similarity
                if similarity >= query.min_similarity:
                    candidates.append((similarity, doc))
            
            # Sort by similarity (descending) and limit
            candidates.sort(key=lambda x: x[0], reverse=True)
            candidates = candidates[:query.limit]
            
            # Convert to results
            results = []
            for similarity, doc in candidates:
                result = VectorSearchResult(
                    id=doc["_id"],
                    score=float(similarity),
                    text=doc["text"],
                    metadata=doc.get("metadata", {})
                )
                
                if query.include_metadata:
                    result.metadata.update({
                        "language": doc.get("language"),
                        "embedding_model": doc.get("embedding_model"), 
                        "created_at": doc.get("created_at")
                    })
                
                results.append(result)
            
            logger.debug(f"Native MongoDB search found {len(results)} results from {len(list(cursor))} candidates")
            return results
            
        except Exception as e:
            logger.error(f"Native MongoDB search failed: {e}")
            return []
    
    def get_embedding(self, id: str) -> Optional[Dict[str, Any]]:
        """Get embedding by ID"""
        try:
            doc = self.collection.find_one({"_id": id})
            return doc
        except Exception as e:
            logger.error(f"Failed to get embedding {id}: {e}")
            return None
    
    def delete_embedding(self, id: str) -> bool:
        """Delete embedding by ID"""
        try:
            result = self.collection.delete_one({"_id": id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete embedding {id}: {e}")
            return False
    
    def count_embeddings(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count embeddings with optional filters"""
        try:
            return self.collection.count_documents(filters or {})
        except Exception as e:
            logger.error(f"Failed to count embeddings: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            stats = {
                "total_embeddings": self.collection.count_documents({}),
                "collection_name": self.config.collection_name,
                "similarity_metric": self.config.similarity_metric.value if hasattr(self.config.similarity_metric, 'value') else str(self.config.similarity_metric),
                "dimension": self.config.dimension,
                "atlas_vector_search": self.config.use_atlas_vector_search
            }
            
            # Language distribution
            pipeline = [
                {"$group": {"_id": "$language", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            language_dist = list(self.collection.aggregate(pipeline))
            stats["language_distribution"] = {
                item["_id"] or "unknown": item["count"] 
                for item in language_dist
            }
            
            # Model distribution
            pipeline = [
                {"$group": {"_id": "$embedding_model", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            model_dist = list(self.collection.aggregate(pipeline))
            stats["model_distribution"] = {
                item["_id"]: item["count"] 
                for item in model_dist
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}
    
    async def store_embedding_async(self, 
                                   id: str,
                                   embedding_result: EmbeddingResult,
                                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Async version of store_embedding"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.store_embedding,
            id,
            embedding_result,
            metadata
        )
    
    async def search_similar_async(self, query: SearchQuery) -> List[VectorSearchResult]:
        """Async version of search_similar"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.search_similar,
            query
        )
    
    def clear_collection(self):
        """Clear all embeddings (use with caution)"""
        try:
            result = self.collection.delete_many({})
            logger.warning(f"Cleared {result.deleted_count} embeddings from collection")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return 0