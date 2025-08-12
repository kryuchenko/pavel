"""
Semantic anomaly detection for Google Play reviews.

Uses embedding vectors and semantic similarity to detect outliers
in the semantic space - reviews that are semantically different
from the typical patterns.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from collections import defaultdict, Counter
import math

from pavel.core.logger import get_logger

logger = get_logger(__name__)

class SemanticMethod(Enum):
    """Semantic anomaly detection methods"""
    COSINE_DISTANCE = "cosine_distance"
    EUCLIDEAN_DISTANCE = "euclidean_distance" 
    MAHALANOBIS_DISTANCE = "mahalanobis_distance"
    SEMANTIC_DENSITY = "semantic_density"
    EMBEDDING_CLUSTERING = "embedding_clustering"
    SEMANTIC_DRIFT = "semantic_drift"
    CONTENT_DIVERGENCE = "content_divergence"

@dataclass
class SemanticConfig:
    """Configuration for semantic anomaly detection"""
    # Distance thresholds
    cosine_threshold: float = 0.7  # Reviews with cosine similarity < 0.3 to cluster center
    euclidean_threshold: float = 2.0
    
    # Density-based parameters
    density_radius: float = 0.5
    min_density_neighbors: int = 5
    
    # Clustering parameters
    embedding_eps: float = 0.3
    embedding_min_samples: int = 3
    
    # Semantic drift parameters
    drift_window_size: int = 100
    drift_threshold: float = 0.15
    
    # Content divergence parameters
    divergence_top_k: int = 10
    divergence_threshold: float = 0.8
    
    # Methods to apply
    methods: List[SemanticMethod] = None
    
    # Minimum sample size
    min_sample_size: int = 10
    
    # Use existing embeddings or generate new ones
    use_existing_embeddings: bool = True
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = [
                SemanticMethod.COSINE_DISTANCE,
                SemanticMethod.SEMANTIC_DENSITY,
                SemanticMethod.EMBEDDING_CLUSTERING
            ]

@dataclass
class SemanticAnomaly:
    """Semantic anomaly detection result"""
    review_id: str
    app_id: str
    method: SemanticMethod
    score: float
    is_anomaly: bool
    embedding: Optional[np.ndarray] = None
    nearest_neighbors: Optional[List[str]] = None
    cluster_center: Optional[np.ndarray] = None
    semantic_topics: Optional[List[str]] = None
    explanation: Optional[str] = None

class SemanticAnomalyDetector:
    """
    Semantic anomaly detector for Google Play reviews.
    
    Detects outliers in the semantic embedding space using:
    - Distance-based methods (cosine, euclidean, mahalanobis)
    - Density-based outlier detection in embedding space
    - Semantic clustering of embeddings (DBSCAN on embeddings)
    - Temporal semantic drift detection
    - Content divergence analysis
    
    Requires embeddings from Stage 4 (Vector Embeddings).
    """
    
    def __init__(self, config: Optional[SemanticConfig] = None):
        self.config = config or SemanticConfig()
        
        # Will be injected from embedding pipeline
        self.embedding_generator = None
        self.vector_store = None
        
    def set_embedding_components(self, embedding_generator, vector_store):
        """Set embedding components from Stage 4"""
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
    
    def detect_anomalies(self, reviews: List[Dict[str, Any]], 
                        embeddings: Optional[List[np.ndarray]] = None) -> List[SemanticAnomaly]:
        """
        Detect semantic anomalies in reviews.
        
        Args:
            reviews: List of review dictionaries
            embeddings: Optional precomputed embeddings
            
        Returns:
            List of detected semantic anomalies
        """
        if len(reviews) < self.config.min_sample_size:
            logger.warning(f"Sample size {len(reviews)} below minimum {self.config.min_sample_size}")
            return []
        
        logger.info(f"Analyzing {len(reviews)} reviews for semantic anomalies")
        
        # Get or generate embeddings
        if embeddings is None:
            embeddings = self._get_embeddings(reviews)
        
        if embeddings is None or len(embeddings) == 0:
            logger.error("No embeddings available for semantic analysis")
            return []
        
        logger.debug(f"Using {len(embeddings)} embeddings for semantic analysis")
        
        # Convert to numpy array
        embedding_matrix = np.array(embeddings)
        if embedding_matrix.ndim == 1:
            embedding_matrix = embedding_matrix.reshape(1, -1)
        
        anomalies = []
        
        # Apply each configured method
        for method in self.config.methods:
            try:
                method_anomalies = self._apply_semantic_method(
                    method, embedding_matrix, reviews
                )
                anomalies.extend(method_anomalies)
                logger.debug(f"Method {method.value} found {len(method_anomalies)} anomalies")
            except Exception as e:
                logger.error(f"Semantic method {method.value} failed: {e}")
                continue
        
        # Remove duplicates
        unique_anomalies = self._deduplicate_anomalies(anomalies)
        
        logger.info(f"Found {len(unique_anomalies)} unique semantic anomalies")
        return unique_anomalies
    
    def _get_embeddings(self, reviews: List[Dict[str, Any]]) -> Optional[List[np.ndarray]]:
        """Get embeddings for reviews"""
        if not self.config.use_existing_embeddings or not self.embedding_generator:
            logger.warning("No embedding generator available")
            return None
        
        try:
            embeddings = []
            
            for review in reviews:
                text = review.get('content', '')
                if not text:
                    # Create zero embedding for empty text
                    zero_embedding = np.zeros(384)  # Default E5-small dimension
                    embeddings.append(zero_embedding)
                    continue
                
                # Generate embedding
                embedding_result = self.embedding_generator.generate_single(text)
                embeddings.append(embedding_result.embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            return None
    
    def _apply_semantic_method(self, method: SemanticMethod, embedding_matrix: np.ndarray, 
                              reviews: List[Dict[str, Any]]) -> List[SemanticAnomaly]:
        """Apply specific semantic method"""
        if method == SemanticMethod.COSINE_DISTANCE:
            return self._detect_cosine_anomalies(embedding_matrix, reviews)
        elif method == SemanticMethod.EUCLIDEAN_DISTANCE:
            return self._detect_euclidean_anomalies(embedding_matrix, reviews)
        elif method == SemanticMethod.MAHALANOBIS_DISTANCE:
            return self._detect_mahalanobis_anomalies(embedding_matrix, reviews)
        elif method == SemanticMethod.SEMANTIC_DENSITY:
            return self._detect_density_anomalies(embedding_matrix, reviews)
        elif method == SemanticMethod.EMBEDDING_CLUSTERING:
            return self._detect_clustering_anomalies(embedding_matrix, reviews)
        elif method == SemanticMethod.SEMANTIC_DRIFT:
            return self._detect_drift_anomalies(embedding_matrix, reviews)
        elif method == SemanticMethod.CONTENT_DIVERGENCE:
            return self._detect_divergence_anomalies(embedding_matrix, reviews)
        
        return []
    
    def _detect_cosine_anomalies(self, embedding_matrix: np.ndarray, 
                                reviews: List[Dict[str, Any]]) -> List[SemanticAnomaly]:
        """Detect anomalies using cosine distance from centroid"""
        anomalies = []
        
        # Calculate centroid
        centroid = np.mean(embedding_matrix, axis=0)
        
        for i, embedding in enumerate(embedding_matrix):
            # Calculate cosine similarity to centroid
            cosine_sim = np.dot(embedding, centroid) / (
                np.linalg.norm(embedding) * np.linalg.norm(centroid) + 1e-8
            )
            
            # Check if similarity is below threshold (i.e., dissimilar)
            if cosine_sim < self.config.cosine_threshold:
                review_id = reviews[i].get('review_id', f'review_{i}')
                app_id = reviews[i].get('app_id', '')
                
                anomaly = SemanticAnomaly(
                    review_id=review_id,
                    app_id=app_id,
                    method=SemanticMethod.COSINE_DISTANCE,
                    score=1.0 - cosine_sim,  # Higher score = more anomalous
                    is_anomaly=True,
                    embedding=embedding,
                    cluster_center=centroid,
                    explanation=f"Cosine similarity {cosine_sim:.3f} below threshold {self.config.cosine_threshold}"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_euclidean_anomalies(self, embedding_matrix: np.ndarray, 
                                   reviews: List[Dict[str, Any]]) -> List[SemanticAnomaly]:
        """Detect anomalies using euclidean distance from centroid"""
        anomalies = []
        
        # Calculate centroid
        centroid = np.mean(embedding_matrix, axis=0)
        
        for i, embedding in enumerate(embedding_matrix):
            # Calculate euclidean distance to centroid
            distance = np.linalg.norm(embedding - centroid)
            
            if distance > self.config.euclidean_threshold:
                review_id = reviews[i].get('review_id', f'review_{i}')
                app_id = reviews[i].get('app_id', '')
                
                anomaly = SemanticAnomaly(
                    review_id=review_id,
                    app_id=app_id,
                    method=SemanticMethod.EUCLIDEAN_DISTANCE,
                    score=distance,
                    is_anomaly=True,
                    embedding=embedding,
                    cluster_center=centroid,
                    explanation=f"Euclidean distance {distance:.3f} exceeds threshold {self.config.euclidean_threshold}"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_mahalanobis_anomalies(self, embedding_matrix: np.ndarray, 
                                     reviews: List[Dict[str, Any]]) -> List[SemanticAnomaly]:
        """Detect anomalies using Mahalanobis distance"""
        anomalies = []
        
        try:
            # Calculate covariance matrix
            cov_matrix = np.cov(embedding_matrix.T)
            
            # Check if covariance matrix is invertible
            if np.linalg.det(cov_matrix) == 0:
                logger.warning("Singular covariance matrix, skipping Mahalanobis detection")
                return []
            
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            centroid = np.mean(embedding_matrix, axis=0)
            
            # Calculate threshold (chi-squared distribution)
            # Using 95th percentile of chi-squared distribution
            try:
                from scipy.stats import chi2
                threshold = chi2.ppf(0.95, df=embedding_matrix.shape[1])
            except ImportError:
                # Fallback if scipy not available
                threshold = 2.0 * np.sqrt(embedding_matrix.shape[1])
            
            for i, embedding in enumerate(embedding_matrix):
                diff = embedding - centroid
                mahal_distance = np.sqrt(diff.T @ inv_cov_matrix @ diff)
                
                if mahal_distance > threshold:
                    review_id = reviews[i].get('review_id', f'review_{i}')
                    app_id = reviews[i].get('app_id', '')
                    
                    anomaly = SemanticAnomaly(
                        review_id=review_id,
                        app_id=app_id,
                        method=SemanticMethod.MAHALANOBIS_DISTANCE,
                        score=mahal_distance,
                        is_anomaly=True,
                        embedding=embedding,
                        explanation=f"Mahalanobis distance {mahal_distance:.3f} exceeds threshold {threshold:.3f}"
                    )
                    anomalies.append(anomaly)
            
        except Exception as e:
            logger.warning(f"Mahalanobis distance calculation failed: {e}")
        
        return anomalies
    
    def _detect_density_anomalies(self, embedding_matrix: np.ndarray, 
                                 reviews: List[Dict[str, Any]]) -> List[SemanticAnomaly]:
        """Detect anomalies using local density in embedding space"""
        anomalies = []
        
        for i, embedding in enumerate(embedding_matrix):
            # Count neighbors within radius
            distances = [np.linalg.norm(embedding - other) 
                        for j, other in enumerate(embedding_matrix) if i != j]
            
            neighbors_in_radius = sum(1 for d in distances if d <= self.config.density_radius)
            
            if neighbors_in_radius < self.config.min_density_neighbors:
                review_id = reviews[i].get('review_id', f'review_{i}')
                app_id = reviews[i].get('app_id', '')
                
                # Find nearest neighbors
                sorted_distances = sorted(enumerate(distances), key=lambda x: x[1])
                nearest_neighbors = [
                    reviews[idx].get('review_id', f'review_{idx}') 
                    for idx, _ in sorted_distances[:3]
                ]
                
                anomaly = SemanticAnomaly(
                    review_id=review_id,
                    app_id=app_id,
                    method=SemanticMethod.SEMANTIC_DENSITY,
                    score=self.config.min_density_neighbors - neighbors_in_radius,
                    is_anomaly=True,
                    embedding=embedding,
                    nearest_neighbors=nearest_neighbors,
                    explanation=f"Low density: {neighbors_in_radius} neighbors (min: {self.config.min_density_neighbors})"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_clustering_anomalies(self, embedding_matrix: np.ndarray, 
                                    reviews: List[Dict[str, Any]]) -> List[SemanticAnomaly]:
        """Detect anomalies using DBSCAN clustering on embeddings"""
        anomalies = []
        
        try:
            from sklearn.cluster import DBSCAN
            
            dbscan = DBSCAN(
                eps=self.config.embedding_eps, 
                min_samples=self.config.embedding_min_samples,
                metric='cosine'
            )
            
            cluster_labels = dbscan.fit_predict(embedding_matrix)
            
            for i, label in enumerate(cluster_labels):
                if label == -1:  # Noise point (anomaly)
                    review_id = reviews[i].get('review_id', f'review_{i}')
                    app_id = reviews[i].get('app_id', '')
                    
                    # Find distance to nearest cluster
                    min_dist = float('inf')
                    for j, other_label in enumerate(cluster_labels):
                        if other_label != -1 and i != j:
                            dist = 1 - np.dot(embedding_matrix[i], embedding_matrix[j]) / (
                                np.linalg.norm(embedding_matrix[i]) * np.linalg.norm(embedding_matrix[j]) + 1e-8
                            )
                            min_dist = min(min_dist, dist)
                    
                    anomaly = SemanticAnomaly(
                        review_id=review_id,
                        app_id=app_id,
                        method=SemanticMethod.EMBEDDING_CLUSTERING,
                        score=min_dist if min_dist != float('inf') else 1.0,
                        is_anomaly=True,
                        embedding=embedding_matrix[i],
                        cluster_center=None,
                        explanation=f"DBSCAN outlier in embedding space"
                    )
                    anomalies.append(anomaly)
            
        except ImportError:
            logger.warning("sklearn not available for embedding clustering")
        except Exception as e:
            logger.error(f"Embedding clustering failed: {e}")
        
        return anomalies
    
    def _detect_drift_anomalies(self, embedding_matrix: np.ndarray, 
                               reviews: List[Dict[str, Any]]) -> List[SemanticAnomaly]:
        """Detect semantic drift anomalies over time"""
        anomalies = []
        
        # Sort reviews by timestamp if available
        timestamped_reviews = []
        for i, review in enumerate(reviews):
            timestamp = review.get('created_at')
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        dt = timestamp
                    timestamped_reviews.append((dt, i, embedding_matrix[i]))
                except:
                    continue
        
        if len(timestamped_reviews) < self.config.drift_window_size:
            return anomalies
        
        # Sort by timestamp
        timestamped_reviews.sort(key=lambda x: x[0])
        
        # Analyze drift in sliding windows
        for i in range(len(timestamped_reviews) - self.config.drift_window_size):
            window_embeddings = [item[2] for item in timestamped_reviews[i:i + self.config.drift_window_size]]
            
            # Calculate centroid of current window
            current_centroid = np.mean(window_embeddings, axis=0)
            
            # Compare with next window
            if i + self.config.drift_window_size * 2 <= len(timestamped_reviews):
                next_window_embeddings = [
                    item[2] for item in timestamped_reviews[
                        i + self.config.drift_window_size:i + self.config.drift_window_size * 2
                    ]
                ]
                next_centroid = np.mean(next_window_embeddings, axis=0)
                
                # Calculate drift (cosine distance between centroids)
                cosine_sim = np.dot(current_centroid, next_centroid) / (
                    np.linalg.norm(current_centroid) * np.linalg.norm(next_centroid) + 1e-8
                )
                drift_score = 1.0 - cosine_sim
                
                if drift_score > self.config.drift_threshold:
                    # Mark reviews in the drifted window as anomalous
                    for timestamp, review_idx, embedding in timestamped_reviews[
                        i + self.config.drift_window_size:i + self.config.drift_window_size * 2
                    ]:
                        review_id = reviews[review_idx].get('review_id', f'review_{review_idx}')
                        app_id = reviews[review_idx].get('app_id', '')
                        
                        anomaly = SemanticAnomaly(
                            review_id=review_id,
                            app_id=app_id,
                            method=SemanticMethod.SEMANTIC_DRIFT,
                            score=drift_score,
                            is_anomaly=True,
                            embedding=embedding,
                            explanation=f"Semantic drift detected: {drift_score:.3f} > {self.config.drift_threshold}"
                        )
                        anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_divergence_anomalies(self, embedding_matrix: np.ndarray, 
                                    reviews: List[Dict[str, Any]]) -> List[SemanticAnomaly]:
        """Detect content divergence anomalies"""
        anomalies = []
        
        # Find most representative embeddings (closest to centroid)
        centroid = np.mean(embedding_matrix, axis=0)
        distances_to_centroid = [
            np.linalg.norm(embedding - centroid) 
            for embedding in embedding_matrix
        ]
        
        # Get top-k most representative reviews
        representative_indices = sorted(
            range(len(distances_to_centroid)), 
            key=lambda i: distances_to_centroid[i]
        )[:self.config.divergence_top_k]
        
        representative_embeddings = [embedding_matrix[i] for i in representative_indices]
        representative_centroid = np.mean(representative_embeddings, axis=0)
        
        # Find reviews that diverge significantly from representative content
        for i, embedding in enumerate(embedding_matrix):
            if i in representative_indices:
                continue  # Skip representative reviews
            
            # Calculate similarity to representative centroid
            cosine_sim = np.dot(embedding, representative_centroid) / (
                np.linalg.norm(embedding) * np.linalg.norm(representative_centroid) + 1e-8
            )
            
            if cosine_sim < self.config.divergence_threshold:
                review_id = reviews[i].get('review_id', f'review_{i}')
                app_id = reviews[i].get('app_id', '')
                
                # Extract potential semantic topics (simplified)
                content = reviews[i].get('content', '')
                topics = self._extract_semantic_topics(content)
                
                anomaly = SemanticAnomaly(
                    review_id=review_id,
                    app_id=app_id,
                    method=SemanticMethod.CONTENT_DIVERGENCE,
                    score=1.0 - cosine_sim,
                    is_anomaly=True,
                    embedding=embedding,
                    semantic_topics=topics,
                    explanation=f"Content divergence: {cosine_sim:.3f} < {self.config.divergence_threshold}"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _extract_semantic_topics(self, content: str) -> List[str]:
        """Extract potential semantic topics from content (simplified)"""
        if not content:
            return []
        
        # Simple keyword-based topic extraction
        topics = []
        content_lower = content.lower()
        
        # Technical issues
        if any(word in content_lower for word in ['crash', 'bug', 'error', 'freeze', 'glitch']):
            topics.append('technical_issues')
        
        # UI/UX
        if any(word in content_lower for word in ['interface', 'design', 'ui', 'user', 'experience']):
            topics.append('ui_ux')
        
        # Performance
        if any(word in content_lower for word in ['slow', 'fast', 'speed', 'performance', 'lag']):
            topics.append('performance')
        
        # Features
        if any(word in content_lower for word in ['feature', 'function', 'option', 'setting']):
            topics.append('features')
        
        # Support
        if any(word in content_lower for word in ['support', 'help', 'customer', 'service']):
            topics.append('support')
        
        # General satisfaction
        if any(word in content_lower for word in ['love', 'hate', 'good', 'bad', 'excellent', 'terrible']):
            topics.append('satisfaction')
        
        return topics if topics else ['general']
    
    def _deduplicate_anomalies(self, anomalies: List[SemanticAnomaly]) -> List[SemanticAnomaly]:
        """Remove duplicate anomalies for the same review"""
        seen_reviews = {}
        unique_anomalies = []
        
        # Sort by score (highest first) to keep best anomalies
        anomalies_sorted = sorted(anomalies, key=lambda x: x.score, reverse=True)
        
        for anomaly in anomalies_sorted:
            if anomaly.review_id not in seen_reviews:
                seen_reviews[anomaly.review_id] = anomaly
                unique_anomalies.append(anomaly)
        
        return unique_anomalies
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics"""
        return {
            "config": {
                "methods": [m.value for m in self.config.methods],
                "cosine_threshold": self.config.cosine_threshold,
                "density_radius": self.config.density_radius,
                "drift_threshold": self.config.drift_threshold
            },
            "embedding_generator_available": self.embedding_generator is not None,
            "vector_store_available": self.vector_store is not None
        }