"""
Vector search implementation for semantic similarity search.
Supports MongoDB vector search and in-memory similarity calculations.
"""

import asyncio
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from pymongo import MongoClient
from pymongo.collection import Collection

from pavel.core.config import get_config
from pavel.core.logger import get_logger
from pavel.embeddings.embedding_generator import EmbeddingGenerator

logger = get_logger(__name__)

@dataclass
class SearchResult:
    """Result from vector search."""
    document: Dict
    similarity: float
    score: float  # Review score (1-5)
    content: str
    locale: str
    review_id: str
    app_id: str

@dataclass
class SearchQuery:
    """Vector search query."""
    text: str
    limit: int = 10
    min_similarity: float = 0.0
    filter_params: Optional[Dict] = None  # MongoDB filter
    include_fields: Optional[List[str]] = None

class VectorSearchEngine:
    """
    Vector search engine for semantic similarity search.
    Supports both MongoDB vector search (Atlas) and in-memory similarity calculation.
    """
    
    def __init__(self, collection_name: str, mongo_client: Optional[MongoClient] = None):
        self.config = get_config()
        self.mongo_client = mongo_client or self._get_mongo_client()
        self.db = self.mongo_client[self.config.MONGODB_DATABASE]
        self.collection = self.db[collection_name]
        self.collection_name = collection_name
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()
        
        logger.info(f"Initialized vector search for collection: {collection_name}")
        logger.info(f"Embedding model: {self.embedding_generator.config.model_name}")
    
    def _get_mongo_client(self) -> MongoClient:
        """Get MongoDB client from config."""
        return MongoClient(self.config.MONGODB_URI)
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform vector search for semantic similarity.
        
        Args:
            query: Search query with text and parameters
            
        Returns:
            List of search results sorted by similarity
        """
        logger.info(f"Performing vector search: '{query.text}' (limit: {query.limit})")
        
        # Generate embedding for query
        query_result = await self.embedding_generator.generate_single_async(query.text)
        query_vector = query_result.embedding
        
        try:
            # Try MongoDB Atlas Vector Search first (if available)
            results = await self._atlas_vector_search(query, query_vector)
            logger.info(f"Used MongoDB Atlas Vector Search")
        except Exception as e:
            logger.warning(f"Atlas Vector Search not available: {e}")
            # Fallback to in-memory similarity calculation
            results = await self._memory_vector_search(query, query_vector)
            logger.info(f"Used in-memory vector search")
        
        logger.info(f"Found {len(results)} results")
        return results
    
    async def _atlas_vector_search(self, query: SearchQuery, query_vector: np.ndarray) -> List[SearchResult]:
        """
        Perform vector search using MongoDB Atlas Vector Search.
        Requires MongoDB Atlas with vector search index.
        """
        # MongoDB Atlas $vectorSearch aggregation
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_search_index",  # Vector search index name
                    "path": "embedding.vector",
                    "queryVector": query_vector.tolist(),
                    "numCandidates": query.limit * 10,  # Candidate pool size
                    "limit": query.limit
                }
            }
        ]
        
        # Add filters if specified
        if query.filter_params:
            pipeline.append({"$match": query.filter_params})
        
        # Add score calculation
        pipeline.append({
            "$addFields": {
                "similarity": {"$meta": "vectorSearchScore"}
            }
        })
        
        # Project required fields
        project_fields = {
            "similarity": 1,
            "content": 1,
            "score": 1,
            "locale": 1,
            "reviewId": 1,
            "appId": 1,
            "at": 1,
            "userName": 1,
            "embedding.model": 1
        }
        
        if query.include_fields:
            for field in query.include_fields:
                project_fields[field] = 1
        
        pipeline.append({"$project": project_fields})
        
        # Execute aggregation
        cursor = self.collection.aggregate(pipeline)
        documents = list(cursor)
        
        # Convert to SearchResult objects
        results = []
        for doc in documents:
            if doc.get('similarity', 0) >= query.min_similarity:
                result = SearchResult(
                    document=doc,
                    similarity=doc.get('similarity', 0.0),
                    score=doc.get('score', 0),
                    content=doc.get('content', ''),
                    locale=doc.get('locale', ''),
                    review_id=doc.get('reviewId', ''),
                    app_id=doc.get('appId', '')
                )
                results.append(result)
        
        return results
    
    async def _memory_vector_search(self, query: SearchQuery, query_vector: np.ndarray) -> List[SearchResult]:
        """
        Perform vector search using in-memory similarity calculation.
        Fallback method when Atlas Vector Search is not available.
        """
        # Build MongoDB filter
        mongo_filter = {"embedding": {"$ne": None}}
        if query.filter_params:
            mongo_filter.update(query.filter_params)
        
        # Project fields for efficiency
        projection = {
            "embedding.vector": 1,
            "content": 1,
            "score": 1,
            "locale": 1,
            "reviewId": 1,
            "appId": 1,
            "at": 1,
            "userName": 1
        }
        
        if query.include_fields:
            for field in query.include_fields:
                projection[field] = 1
        
        # Get all documents with embeddings
        cursor = self.collection.find(mongo_filter, projection)
        documents = list(cursor)
        
        logger.info(f"Calculating similarity for {len(documents)} documents")
        
        # Calculate similarities
        similarities = []
        for doc in documents:
            embedding_data = doc.get('embedding')
            if not embedding_data or not embedding_data.get('vector'):
                continue
            
            doc_vector = np.array(embedding_data['vector'])
            similarity = self._cosine_similarity(query_vector, doc_vector)
            
            if similarity >= query.min_similarity:
                similarities.append((similarity, doc))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Take top results
        top_similarities = similarities[:query.limit]
        
        # Convert to SearchResult objects
        results = []
        for similarity, doc in top_similarities:
            result = SearchResult(
                document=doc,
                similarity=similarity,
                score=doc.get('score', 0),
                content=doc.get('content', ''),
                locale=doc.get('locale', ''),
                review_id=doc.get('reviewId', ''),
                app_id=doc.get('appId', '')
            )
            results.append(result)
        
        return results
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    async def find_similar_reviews(self, review_id: str, limit: int = 10) -> List[SearchResult]:
        """
        Find reviews similar to a given review.
        
        Args:
            review_id: ID of the reference review
            limit: Maximum number of results
            
        Returns:
            List of similar reviews
        """
        # Get the reference review
        ref_review = self.collection.find_one({"reviewId": review_id})
        if not ref_review:
            raise ValueError(f"Review not found: {review_id}")
        
        content = ref_review.get('content', '')
        if not content:
            raise ValueError(f"Review has no content: {review_id}")
        
        # Search for similar reviews
        query = SearchQuery(
            text=content,
            limit=limit + 1,  # +1 to exclude the reference review
            filter_params={"reviewId": {"$ne": review_id}}  # Exclude self
        )
        
        results = await self.search(query)
        return results[:limit]  # Return requested limit
    
    async def search_by_sentiment(self, 
                                sentiment: str, 
                                limit: int = 10,
                                locale: Optional[str] = None) -> List[SearchResult]:
        """
        Search reviews by sentiment (positive/negative).
        
        Args:
            sentiment: 'positive' or 'negative'
            limit: Maximum number of results
            locale: Optional language filter
            
        Returns:
            List of reviews matching sentiment
        """
        # Define sentiment queries
        if sentiment.lower() == 'positive':
            query_text = "great excellent awesome amazing wonderful fantastic good"
            score_filter = {"score": {"$gte": 4}}
        elif sentiment.lower() == 'negative':
            query_text = "terrible awful bad horrible disappointing frustrating bugs crashes"
            score_filter = {"score": {"$lte": 2}}
        else:
            raise ValueError("Sentiment must be 'positive' or 'negative'")
        
        # Build filter
        filter_params = score_filter.copy()
        if locale:
            filter_params["locale"] = locale
        
        # Search
        query = SearchQuery(
            text=query_text,
            limit=limit,
            filter_params=filter_params,
            min_similarity=0.3  # Minimum similarity threshold
        )
        
        return await self.search(query)
    
    async def search_issues(self, limit: int = 10, locale: Optional[str] = None) -> List[SearchResult]:
        """
        Search for reviews mentioning technical issues.
        
        Args:
            limit: Maximum number of results
            locale: Optional language filter
            
        Returns:
            List of reviews mentioning issues
        """
        # Multi-language issue keywords
        if locale == 'ru':
            query_text = "баг ошибка вылет проблема не работает глюк лаг тормозит"
        elif locale == 'es':
            query_text = "bug error fallo problema no funciona lento crashea"
        else:
            query_text = "bug error crash freeze lag slow problem issue glitch broken"
        
        filter_params = {}
        if locale:
            filter_params["locale"] = locale
        
        query = SearchQuery(
            text=query_text,
            limit=limit,
            filter_params=filter_params,
            min_similarity=0.4
        )
        
        return await self.search(query)
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        total_reviews = self.collection.count_documents({})
        reviews_with_embeddings = self.collection.count_documents({"embedding": {"$ne": None}})
        
        # Language distribution
        lang_pipeline = [
            {"$group": {"_id": "$locale", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        lang_distribution = list(self.collection.aggregate(lang_pipeline))
        
        # Score distribution  
        score_pipeline = [
            {"$group": {"_id": "$score", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        score_distribution = list(self.collection.aggregate(score_pipeline))
        
        return {
            "collection_name": self.collection_name,
            "total_reviews": total_reviews,
            "reviews_with_embeddings": reviews_with_embeddings,
            "embedding_coverage": reviews_with_embeddings / total_reviews if total_reviews > 0 else 0,
            "language_distribution": lang_distribution,
            "score_distribution": score_distribution
        }
    
    def close(self):
        """Close database connection."""
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("Closed vector search database connection")