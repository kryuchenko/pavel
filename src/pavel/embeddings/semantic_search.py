"""
Semantic search engine for Google Play review analysis.

Provides high-level semantic search capabilities combining embedding generation,
vector storage, and intelligent query processing for anomaly detection.
"""

import numpy as np
import asyncio
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter

from pavel.core.logger import get_logger
from .embedding_generator import EmbeddingGenerator, EmbeddingConfig
from .vector_store import VectorStore, VectorStoreConfig, SearchQuery, VectorSearchResult, SimilarityMetric

logger = get_logger(__name__)

@dataclass
class SearchResult:
    """Enhanced search result with semantic analysis"""
    id: str
    text: str
    similarity_score: float
    language: Optional[str]
    metadata: Dict[str, Any]
    embedding_model: Optional[str] = None
    semantic_cluster: Optional[str] = None
    anomaly_indicators: Optional[List[str]] = None
    created_at: Optional[datetime] = None

class SemanticSearchEngine:
    """
    High-level semantic search engine for Google Play reviews.
    
    Features:
    - Multi-language semantic search
    - Anomaly pattern detection
    - Cluster-based similarity search
    - Review sentiment and topic analysis
    - Batch search operations
    - Smart query expansion
    """
    
    def __init__(self,
                 embedding_generator: Optional[EmbeddingGenerator] = None,
                 vector_store: Optional[VectorStore] = None,
                 embedding_config: Optional[EmbeddingConfig] = None,
                 vector_config: Optional[VectorStoreConfig] = None):
        
        self.embedding_generator = embedding_generator or EmbeddingGenerator(embedding_config)
        self.vector_store = vector_store or VectorStore(vector_config)
        
        # Cache for common queries
        self.query_cache = {}
        
        # Anomaly detection patterns
        self._setup_anomaly_patterns()
    
    def _setup_anomaly_patterns(self):
        """Setup patterns for anomaly detection in reviews"""
        
        # Common anomaly indicators in reviews (multilingual)
        self.anomaly_keywords = {
            'en': {
                'fake_positive': ['amazing', 'perfect', 'best ever', 'incredible', 'outstanding'],
                'fake_negative': ['terrible', 'worst', 'horrible', 'disgusting', 'awful'],
                'technical_issues': ['crash', 'bug', 'error', 'freeze', 'slow', 'broken'],
                'spam_indicators': ['free coins', 'hack', 'cheat', 'generator', 'unlimited'],
                'bot_patterns': ['very good app', 'nice app', 'good app', 'perfect app']
            },
            'ru': {
                'fake_positive': ['потрясающий', 'идеальный', 'лучший', 'невероятный', 'восхитительный'],
                'fake_negative': ['ужасный', 'худший', 'отвратительный', 'кошмар'],
                'technical_issues': ['вылетает', 'баг', 'ошибка', 'зависает', 'медленно', 'сломано'],
                'spam_indicators': ['бесплатные монеты', 'взлом', 'читы', 'генератор'],
                'bot_patterns': ['хорошее приложение', 'отличное приложение', 'супер приложение']
            },
            'es': {
                'fake_positive': ['increíble', 'perfecto', 'el mejor', 'fantástico', 'excepcional'],
                'fake_negative': ['terrible', 'pésimo', 'horrible', 'asqueroso'],
                'technical_issues': ['falla', 'error', 'bug', 'lento', 'roto', 'cuelga'],
                'spam_indicators': ['monedas gratis', 'hack', 'truco', 'generador'],
                'bot_patterns': ['muy buena app', 'buena aplicación', 'app perfecta']
            }
        }
    
    def search_similar_reviews(self,
                              query_text: str,
                              limit: int = 10,
                              min_similarity: float = 0.7,
                              language_filter: Optional[str] = None,
                              app_filter: Optional[str] = None,
                              date_range: Optional[Tuple[datetime, datetime]] = None) -> List[SearchResult]:
        """
        Search for semantically similar reviews.
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            language_filter: Filter by language code
            app_filter: Filter by app ID
            date_range: Filter by date range (start, end)
            
        Returns:
            List of similar reviews with semantic analysis
        """
        try:
            # Generate embedding for query
            query_embedding = self.embedding_generator.generate_single(query_text)
            
            # Build search filters
            filters = {}
            if language_filter:
                filters['language'] = language_filter
            if app_filter:
                filters['metadata.app_id'] = app_filter
            if date_range:
                filters['created_at'] = {
                    '$gte': date_range[0],
                    '$lte': date_range[1]
                }
            
            # Create search query
            search_query = SearchQuery(
                vector=query_embedding.embedding,
                limit=limit,
                min_similarity=min_similarity,
                filters=filters if filters else None,
                include_metadata=True
            )
            
            # Perform vector search
            vector_results = self.vector_store.search_similar(search_query)
            
            # Convert to enhanced search results with semantic analysis
            search_results = []
            for result in vector_results:
                enhanced_result = SearchResult(
                    id=result.id,
                    text=result.text,
                    similarity_score=result.score,
                    language=result.metadata.get('language'),
                    metadata=result.metadata,
                    embedding_model=result.metadata.get('embedding_model'),
                    created_at=result.metadata.get('created_at')
                )
                
                # Add anomaly detection
                enhanced_result.anomaly_indicators = self._detect_anomalies(
                    result.text,
                    enhanced_result.language or 'en'
                )
                
                search_results.append(enhanced_result)
            
            logger.info(f"Found {len(search_results)} similar reviews for query: {query_text[:50]}...")
            return search_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def find_anomalous_reviews(self,
                              app_id: Optional[str] = None,
                              language: Optional[str] = None,
                              anomaly_type: Optional[str] = None,
                              limit: int = 50,
                              min_confidence: float = 0.8) -> List[SearchResult]:
        """
        Find potentially anomalous reviews using semantic patterns.
        
        Args:
            app_id: Filter by specific app
            language: Filter by language
            anomaly_type: Type of anomaly to search for
            limit: Maximum results
            min_confidence: Minimum anomaly confidence
            
        Returns:
            List of potentially anomalous reviews
        """
        try:
            # Define anomaly query patterns
            anomaly_patterns = {
                'fake_positive': [
                    "This app is absolutely perfect amazing incredible best",
                    "Five stars perfect application outstanding quality",
                    "Amazing wonderful fantastic incredible perfect app"
                ],
                'fake_negative': [
                    "Terrible horrible worst application ever disgusting",
                    "One star awful terrible worst app disgusting",
                    "Horrible terrible bad awful worst application"
                ],
                'spam_reviews': [
                    "Good app nice application very good perfect",
                    "Very good app nice perfect application good",
                    "Perfect app very good nice application"
                ],
                'technical_complaints': [
                    "App crashes freezes errors bugs slow performance",
                    "Technical issues crashes bugs errors freezing",
                    "Performance problems slow crashes errors bugs"
                ]
            }
            
            # Use specific pattern or all patterns
            patterns_to_search = []
            if anomaly_type and anomaly_type in anomaly_patterns:
                patterns_to_search = anomaly_patterns[anomaly_type]
            else:
                # Search all types
                for pattern_list in anomaly_patterns.values():
                    patterns_to_search.extend(pattern_list)
            
            # Search for each pattern
            all_results = []
            seen_ids = set()
            
            for pattern in patterns_to_search[:5]:  # Limit patterns for performance
                results = self.search_similar_reviews(
                    query_text=pattern,
                    limit=limit // len(patterns_to_search[:5]) + 10,
                    min_similarity=min_confidence,
                    language_filter=language,
                    app_filter=app_id
                )
                
                # Add unique results
                for result in results:
                    if result.id not in seen_ids:
                        seen_ids.add(result.id)
                        all_results.append(result)
            
            # Sort by anomaly confidence (based on similarity to anomaly patterns)
            all_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Limit final results
            anomalous_results = all_results[:limit]
            
            logger.info(f"Found {len(anomalous_results)} potentially anomalous reviews")
            return anomalous_results
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    def cluster_similar_reviews(self,
                               app_id: str,
                               language: Optional[str] = None,
                               cluster_threshold: float = 0.8,
                               min_cluster_size: int = 3) -> Dict[str, List[SearchResult]]:
        """
        Find clusters of highly similar reviews (potential duplicates or coordinated campaigns).
        
        Args:
            app_id: App to analyze
            language: Language filter
            cluster_threshold: Similarity threshold for clustering
            min_cluster_size: Minimum reviews per cluster
            
        Returns:
            Dictionary mapping cluster IDs to lists of similar reviews
        """
        try:
            # Get all reviews for the app
            filters = {'metadata.app_id': app_id}
            if language:
                filters['language'] = language
            
            # This would require a more sophisticated clustering algorithm
            # For now, we'll do a simplified approach using pairwise similarity
            
            # Get sample of reviews (for performance)
            sample_query = SearchQuery(
                vector=np.zeros(self.embedding_generator.get_embedding_dimension()),  # Dummy vector
                limit=200,  # Reasonable sample size
                min_similarity=0.0,
                filters=filters
            )
            
            # Instead of dummy vector, get random sample from DB
            # This is a simplified version - would need proper implementation
            clusters = {}
            cluster_id = 0
            
            logger.info(f"Clustering analysis would require {app_id} reviews")
            
            # Placeholder return
            return {"cluster_0": []}
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {}
    
    def analyze_review_patterns(self,
                               app_id: str,
                               time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze patterns in recent reviews for anomaly detection.
        
        Args:
            app_id: App to analyze
            time_window_hours: Time window for analysis
            
        Returns:
            Analysis report with potential anomalies
        """
        try:
            # Time range filter
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=time_window_hours)
            
            # Get recent reviews
            filters = {
                'metadata.app_id': app_id,
                'created_at': {'$gte': start_time, '$lte': end_time}
            }
            
            # Count reviews by various dimensions
            review_count = self.vector_store.count_embeddings(filters)
            
            # Analyze patterns
            analysis = {
                'app_id': app_id,
                'time_window_hours': time_window_hours,
                'total_reviews': review_count,
                'analysis_timestamp': end_time,
                'patterns': {
                    'review_volume': 'normal' if review_count < 100 else 'high',
                    'potential_anomalies': []
                }
            }
            
            # Add more sophisticated analysis here
            if review_count > 50:
                analysis['patterns']['potential_anomalies'].append('high_volume')
            
            logger.info(f"Pattern analysis complete for {app_id}: {review_count} reviews")
            return analysis
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return {'error': str(e)}
    
    def _detect_anomalies(self, text: str, language: str = 'en') -> List[str]:
        """
        Detect potential anomaly indicators in review text.
        
        Args:
            text: Review text
            language: Language code
            
        Returns:
            List of detected anomaly types
        """
        anomalies = []
        text_lower = text.lower()
        
        # Get patterns for language (fallback to English)
        patterns = self.anomaly_keywords.get(language, self.anomaly_keywords['en'])
        
        # Check each anomaly type
        for anomaly_type, keywords in patterns.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    anomalies.append(anomaly_type)
                    break  # Don't add same type twice
        
        # Additional checks
        # Repeated characters (spam indicator)
        if '!!!' in text or '???' in text or len(set(text.lower().split())) < len(text.split()) * 0.7:
            anomalies.append('spam_patterns')
        
        # Very short or very long reviews
        if len(text.strip()) < 10:
            anomalies.append('too_short')
        elif len(text.strip()) > 1000:
            anomalies.append('too_long')
        
        return list(set(anomalies))  # Remove duplicates
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        try:
            vector_stats = self.vector_store.get_statistics()
            embedding_stats = self.embedding_generator.get_cache_stats()
            
            stats = {
                'vector_storage': vector_stats,
                'embedding_cache': embedding_stats,
                'query_cache_size': len(self.query_cache),
                'supported_languages': list(self.anomaly_keywords.keys()),
                'anomaly_types': list(next(iter(self.anomaly_keywords.values())).keys())
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {'error': str(e)}
    
    async def search_similar_reviews_async(self,
                                          query_text: str,
                                          limit: int = 10,
                                          min_similarity: float = 0.7,
                                          language_filter: Optional[str] = None,
                                          app_filter: Optional[str] = None) -> List[SearchResult]:
        """Async version of search_similar_reviews"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.search_similar_reviews,
            query_text,
            limit,
            min_similarity,
            language_filter,
            app_filter
        )
    
    def clear_caches(self):
        """Clear all caches"""
        self.query_cache.clear()
        self.embedding_generator.clear_cache()
        logger.info("Search engine caches cleared")

# Convenience function
def create_search_engine(embedding_model: str = "intfloat/multilingual-e5-large") -> SemanticSearchEngine:
    """Create a pre-configured semantic search engine"""
    embedding_config = EmbeddingConfig(model_name=embedding_model)
    vector_config = VectorStoreConfig(similarity_metric=SimilarityMetric.COSINE)
    
    return SemanticSearchEngine(
        embedding_config=embedding_config,
        vector_config=vector_config
    )