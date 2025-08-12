"""
Clustering-based anomaly detection for Google Play reviews.

Uses machine learning clustering algorithms (DBSCAN, Isolation Forest, etc.) 
to identify outliers and anomalous patterns in review data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from collections import defaultdict, Counter
import warnings

from pavel.core.logger import get_logger

logger = get_logger(__name__)

class ClusteringMethod(Enum):
    """Clustering-based anomaly detection methods"""
    DBSCAN = "dbscan"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    ONE_CLASS_SVM = "one_class_svm"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    KMEANS_DISTANCE = "kmeans_distance"

@dataclass
class ClusteringConfig:
    """Configuration for clustering anomaly detection"""
    # DBSCAN parameters
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    
    # Isolation Forest parameters
    isolation_contamination: float = 0.1
    isolation_n_estimators: int = 100
    isolation_max_samples: str = "auto"
    
    # LOF parameters
    lof_n_neighbors: int = 20
    lof_contamination: float = 0.1
    
    # One-Class SVM parameters
    svm_nu: float = 0.1
    svm_kernel: str = "rbf"
    svm_gamma: str = "scale"
    
    # Gaussian Mixture parameters
    gmm_n_components: int = 3
    gmm_covariance_type: str = "full"
    
    # K-means parameters
    kmeans_n_clusters: int = 5
    kmeans_distance_threshold: float = 2.0
    
    # Feature selection
    use_text_features: bool = True
    use_rating_features: bool = True
    use_temporal_features: bool = True
    use_linguistic_features: bool = True
    
    # Methods to apply
    methods: List[ClusteringMethod] = None
    
    # Minimum sample size
    min_sample_size: int = 20
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = [
                ClusteringMethod.DBSCAN,
                ClusteringMethod.ISOLATION_FOREST,
                ClusteringMethod.LOCAL_OUTLIER_FACTOR
            ]

@dataclass
class ClusteringAnomaly:
    """Clustering anomaly detection result"""
    review_id: str
    app_id: str
    method: ClusteringMethod
    score: float
    is_anomaly: bool
    cluster_id: Optional[int] = None
    distance_to_cluster: Optional[float] = None
    nearest_neighbors: Optional[List[str]] = None
    feature_contributions: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None

class ClusteringAnomalyDetector:
    """
    Clustering-based anomaly detector for Google Play reviews.
    
    Uses various clustering algorithms to identify outliers:
    - DBSCAN for density-based outliers
    - Isolation Forest for general anomalies
    - Local Outlier Factor for local density anomalies
    - One-Class SVM for boundary detection
    - Gaussian Mixture for probability-based detection
    - K-means distance for centroid-based detection
    
    Features analyzed:
    - Text characteristics (length, complexity, linguistic patterns)
    - Rating patterns
    - Temporal patterns
    - Linguistic features (language, sentiment indicators)
    """
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        self.config = config or ClusteringConfig()
        self._sklearn_available = self._check_sklearn()
        
        if not self._sklearn_available:
            logger.error("Clustering detection requires sklearn. Install with: pip install scikit-learn")
            self.config.methods = []
        
        # Initialize feature extractors
        self._feature_cache = {}
    
    def _check_sklearn(self) -> bool:
        """Check if sklearn is available"""
        try:
            import sklearn
            return True
        except ImportError:
            return False
    
    def detect_anomalies(self, reviews: List[Dict[str, Any]]) -> List[ClusteringAnomaly]:
        """
        Detect clustering-based anomalies in reviews.
        
        Args:
            reviews: List of review dictionaries
            
        Returns:
            List of detected anomalies
        """
        if not self._sklearn_available:
            logger.error("sklearn not available for clustering detection")
            return []
        
        if len(reviews) < self.config.min_sample_size:
            logger.warning(f"Sample size {len(reviews)} below minimum {self.config.min_sample_size}")
            return []
        
        logger.info(f"Analyzing {len(reviews)} reviews for clustering anomalies")
        
        # Extract features
        feature_matrix, feature_names, review_ids = self._extract_features(reviews)
        
        if feature_matrix is None or feature_matrix.shape[1] == 0:
            logger.error("No features extracted for clustering analysis")
            return []
        
        logger.debug(f"Extracted feature matrix: {feature_matrix.shape}")
        
        anomalies = []
        
        # Apply each configured method
        for method in self.config.methods:
            try:
                method_anomalies = self._apply_clustering_method(
                    method, feature_matrix, feature_names, review_ids, reviews
                )
                anomalies.extend(method_anomalies)
                logger.debug(f"Method {method.value} found {len(method_anomalies)} anomalies")
            except Exception as e:
                logger.error(f"Clustering method {method.value} failed: {e}")
                continue
        
        # Remove duplicates
        unique_anomalies = self._deduplicate_anomalies(anomalies)
        
        logger.info(f"Found {len(unique_anomalies)} unique clustering anomalies")
        return unique_anomalies
    
    def _extract_features(self, reviews: List[Dict[str, Any]]) -> Tuple[Optional[np.ndarray], List[str], List[str]]:
        """Extract feature matrix from reviews"""
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            features = []
            feature_names = []
            review_ids = []
            
            for review in reviews:
                review_id = review.get('review_id', str(len(review_ids)))
                review_ids.append(review_id)
                
                feature_vector = []
                
                # Text features
                if self.config.use_text_features:
                    text_features = self._extract_text_features(review)
                    feature_vector.extend(text_features)
                    if len(feature_names) < len(feature_vector):
                        feature_names.extend([
                            'text_length', 'word_count', 'avg_word_length', 
                            'sentence_count', 'char_density', 'uppercase_ratio',
                            'punctuation_ratio', 'digit_ratio', 'special_char_ratio'
                        ])
                
                # Rating features
                if self.config.use_rating_features:
                    rating_features = self._extract_rating_features(review)
                    feature_vector.extend(rating_features)
                    if len(feature_names) < len(feature_vector):
                        feature_names.extend(['rating', 'rating_normalized'])
                
                # Temporal features
                if self.config.use_temporal_features:
                    temporal_features = self._extract_temporal_features(review)
                    feature_vector.extend(temporal_features)
                    if len(feature_names) < len(feature_vector):
                        feature_names.extend([
                            'hour_of_day', 'day_of_week', 'is_weekend', 'is_holiday_season'
                        ])
                
                # Linguistic features
                if self.config.use_linguistic_features:
                    linguistic_features = self._extract_linguistic_features(review)
                    feature_vector.extend(linguistic_features)
                    if len(feature_names) < len(feature_vector):
                        feature_names.extend([
                            'language_confidence', 'sentiment_polarity', 'has_emoji',
                            'has_urls', 'has_mentions', 'repetition_ratio'
                        ])
                
                features.append(feature_vector)
            
            if not features:
                return None, [], []
            
            # Convert to numpy array
            feature_matrix = np.array(features, dtype=np.float32)
            
            # Handle NaN values
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Standardize features
            if feature_matrix.shape[0] > 1 and feature_matrix.shape[1] > 0:
                scaler = StandardScaler()
                feature_matrix = scaler.fit_transform(feature_matrix)
            
            return feature_matrix, feature_names[:feature_matrix.shape[1]], review_ids
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None, [], []
    
    def _extract_text_features(self, review: Dict[str, Any]) -> List[float]:
        """Extract text-based features"""
        content = review.get('content', '')
        if not content:
            return [0.0] * 9
        
        # Basic text statistics
        text_length = len(content)
        words = content.split()
        word_count = len(words)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        sentence_count = len([s for s in content.split('.') if s.strip()])
        
        # Character analysis
        char_density = text_length / word_count if word_count > 0 else 0
        alpha_chars = sum(1 for c in content if c.isalpha())
        uppercase_ratio = sum(1 for c in content if c.isupper()) / alpha_chars if alpha_chars > 0 else 0
        punctuation_ratio = sum(1 for c in content if c in '.,!?;:()[]{}') / text_length if text_length > 0 else 0
        digit_ratio = sum(1 for c in content if c.isdigit()) / text_length if text_length > 0 else 0
        special_char_ratio = sum(1 for c in content if not c.isalnum() and not c.isspace()) / text_length if text_length > 0 else 0
        
        return [
            text_length, word_count, avg_word_length, sentence_count,
            char_density, uppercase_ratio, punctuation_ratio, 
            digit_ratio, special_char_ratio
        ]
    
    def _extract_rating_features(self, review: Dict[str, Any]) -> List[float]:
        """Extract rating-based features"""
        rating = review.get('rating', 0)
        rating_normalized = (rating - 3.0) / 2.0 if rating > 0 else 0.0  # Normalize to [-1, 1]
        
        return [float(rating), rating_normalized]
    
    def _extract_temporal_features(self, review: Dict[str, Any]) -> List[float]:
        """Extract temporal features"""
        timestamp = review.get('created_at')
        if not timestamp:
            return [0.0] * 4
        
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif isinstance(timestamp, datetime):
                dt = timestamp
            else:
                return [0.0] * 4
            
            hour_of_day = dt.hour / 24.0  # Normalize to [0, 1]
            day_of_week = dt.weekday() / 6.0  # Normalize to [0, 1]
            is_weekend = float(dt.weekday() >= 5)
            is_holiday_season = float(dt.month in [12, 1])  # December/January
            
            return [hour_of_day, day_of_week, is_weekend, is_holiday_season]
            
        except Exception:
            return [0.0] * 4
    
    def _extract_linguistic_features(self, review: Dict[str, Any]) -> List[float]:
        """Extract linguistic features"""
        content = review.get('content', '')
        if not content:
            return [0.0] * 6
        
        # Language confidence (mock - would use real language detection)
        language_confidence = 1.0  # Assume high confidence for now
        
        # Simple sentiment analysis (mock)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'worst']
        
        content_lower = content.lower()
        pos_count = sum(1 for word in positive_words if word in content_lower)
        neg_count = sum(1 for word in negative_words if word in content_lower)
        sentiment_polarity = (pos_count - neg_count) / len(content.split()) if content.split() else 0.0
        
        # Pattern detection
        has_emoji = float(any(ord(char) > 127 for char in content))
        has_urls = float('http' in content.lower() or 'www.' in content.lower())
        has_mentions = float('@' in content)
        
        # Repetition analysis
        words = content.split()
        if words:
            word_counts = Counter(words)
            repetition_ratio = sum(1 for count in word_counts.values() if count > 1) / len(words)
        else:
            repetition_ratio = 0.0
        
        return [
            language_confidence, sentiment_polarity, has_emoji,
            has_urls, has_mentions, repetition_ratio
        ]
    
    def _apply_clustering_method(self, method: ClusteringMethod, feature_matrix: np.ndarray, 
                                feature_names: List[str], review_ids: List[str], 
                                reviews: List[Dict[str, Any]]) -> List[ClusteringAnomaly]:
        """Apply specific clustering method"""
        if method == ClusteringMethod.DBSCAN:
            return self._detect_dbscan_anomalies(feature_matrix, feature_names, review_ids, reviews)
        elif method == ClusteringMethod.ISOLATION_FOREST:
            return self._detect_isolation_forest_anomalies(feature_matrix, feature_names, review_ids, reviews)
        elif method == ClusteringMethod.LOCAL_OUTLIER_FACTOR:
            return self._detect_lof_anomalies(feature_matrix, feature_names, review_ids, reviews)
        elif method == ClusteringMethod.ONE_CLASS_SVM:
            return self._detect_svm_anomalies(feature_matrix, feature_names, review_ids, reviews)
        elif method == ClusteringMethod.GAUSSIAN_MIXTURE:
            return self._detect_gmm_anomalies(feature_matrix, feature_names, review_ids, reviews)
        elif method == ClusteringMethod.KMEANS_DISTANCE:
            return self._detect_kmeans_anomalies(feature_matrix, feature_names, review_ids, reviews)
        
        return []
    
    def _detect_dbscan_anomalies(self, feature_matrix: np.ndarray, feature_names: List[str], 
                                review_ids: List[str], reviews: List[Dict[str, Any]]) -> List[ClusteringAnomaly]:
        """Detect anomalies using DBSCAN"""
        try:
            from sklearn.cluster import DBSCAN
            
            dbscan = DBSCAN(eps=self.config.dbscan_eps, min_samples=self.config.dbscan_min_samples)
            cluster_labels = dbscan.fit_predict(feature_matrix)
            
            anomalies = []
            for i, label in enumerate(cluster_labels):
                if label == -1:  # Noise point (anomaly)
                    app_id = reviews[i].get('app_id', '')
                    
                    # Calculate distance to nearest cluster
                    distances = []
                    for j, other_label in enumerate(cluster_labels):
                        if other_label != -1 and i != j:
                            dist = np.linalg.norm(feature_matrix[i] - feature_matrix[j])
                            distances.append(dist)
                    
                    min_distance = min(distances) if distances else 0.0
                    
                    anomaly = ClusteringAnomaly(
                        review_id=review_ids[i],
                        app_id=app_id,
                        method=ClusteringMethod.DBSCAN,
                        score=min_distance,
                        is_anomaly=True,
                        cluster_id=-1,
                        distance_to_cluster=min_distance,
                        explanation=f"DBSCAN outlier with distance {min_distance:.3f} to nearest cluster"
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"DBSCAN detection failed: {e}")
            return []
    
    def _detect_isolation_forest_anomalies(self, feature_matrix: np.ndarray, feature_names: List[str], 
                                          review_ids: List[str], reviews: List[Dict[str, Any]]) -> List[ClusteringAnomaly]:
        """Detect anomalies using Isolation Forest"""
        try:
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(
                contamination=self.config.isolation_contamination,
                n_estimators=self.config.isolation_n_estimators,
                max_samples=self.config.isolation_max_samples,
                random_state=42
            )
            
            predictions = iso_forest.fit_predict(feature_matrix)
            anomaly_scores = iso_forest.decision_function(feature_matrix)
            
            anomalies = []
            for i, (prediction, score) in enumerate(zip(predictions, anomaly_scores)):
                if prediction == -1:  # Anomaly
                    app_id = reviews[i].get('app_id', '')
                    
                    anomaly = ClusteringAnomaly(
                        review_id=review_ids[i],
                        app_id=app_id,
                        method=ClusteringMethod.ISOLATION_FOREST,
                        score=abs(score),
                        is_anomaly=True,
                        explanation=f"Isolation Forest anomaly score: {score:.3f}"
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Isolation Forest detection failed: {e}")
            return []
    
    def _detect_lof_anomalies(self, feature_matrix: np.ndarray, feature_names: List[str], 
                             review_ids: List[str], reviews: List[Dict[str, Any]]) -> List[ClusteringAnomaly]:
        """Detect anomalies using Local Outlier Factor"""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            lof = LocalOutlierFactor(
                n_neighbors=min(self.config.lof_n_neighbors, len(feature_matrix) - 1),
                contamination=self.config.lof_contamination
            )
            
            predictions = lof.fit_predict(feature_matrix)
            negative_outlier_factors = lof.negative_outlier_factor_
            
            anomalies = []
            for i, (prediction, score) in enumerate(zip(predictions, negative_outlier_factors)):
                if prediction == -1:  # Anomaly
                    app_id = reviews[i].get('app_id', '')
                    
                    anomaly = ClusteringAnomaly(
                        review_id=review_ids[i],
                        app_id=app_id,
                        method=ClusteringMethod.LOCAL_OUTLIER_FACTOR,
                        score=abs(score),
                        is_anomaly=True,
                        explanation=f"LOF outlier factor: {score:.3f}"
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"LOF detection failed: {e}")
            return []
    
    def _detect_svm_anomalies(self, feature_matrix: np.ndarray, feature_names: List[str], 
                             review_ids: List[str], reviews: List[Dict[str, Any]]) -> List[ClusteringAnomaly]:
        """Detect anomalies using One-Class SVM"""
        try:
            from sklearn.svm import OneClassSVM
            
            svm = OneClassSVM(
                nu=self.config.svm_nu,
                kernel=self.config.svm_kernel,
                gamma=self.config.svm_gamma
            )
            
            predictions = svm.fit_predict(feature_matrix)
            decision_scores = svm.decision_function(feature_matrix)
            
            anomalies = []
            for i, (prediction, score) in enumerate(zip(predictions, decision_scores)):
                if prediction == -1:  # Anomaly
                    app_id = reviews[i].get('app_id', '')
                    
                    anomaly = ClusteringAnomaly(
                        review_id=review_ids[i],
                        app_id=app_id,
                        method=ClusteringMethod.ONE_CLASS_SVM,
                        score=abs(score),
                        is_anomaly=True,
                        explanation=f"One-Class SVM decision score: {score:.3f}"
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"One-Class SVM detection failed: {e}")
            return []
    
    def _detect_gmm_anomalies(self, feature_matrix: np.ndarray, feature_names: List[str], 
                             review_ids: List[str], reviews: List[Dict[str, Any]]) -> List[ClusteringAnomaly]:
        """Detect anomalies using Gaussian Mixture Model"""
        try:
            from sklearn.mixture import GaussianMixture
            
            gmm = GaussianMixture(
                n_components=min(self.config.gmm_n_components, len(feature_matrix) // 5),
                covariance_type=self.config.gmm_covariance_type,
                random_state=42
            )
            
            gmm.fit(feature_matrix)
            log_probabilities = gmm.score_samples(feature_matrix)
            
            # Use low probability as anomaly indicator
            threshold = np.percentile(log_probabilities, self.config.lof_contamination * 100)
            
            anomalies = []
            for i, log_prob in enumerate(log_probabilities):
                if log_prob < threshold:  # Low probability = anomaly
                    app_id = reviews[i].get('app_id', '')
                    
                    anomaly = ClusteringAnomaly(
                        review_id=review_ids[i],
                        app_id=app_id,
                        method=ClusteringMethod.GAUSSIAN_MIXTURE,
                        score=abs(log_prob),
                        is_anomaly=True,
                        explanation=f"GMM log probability: {log_prob:.3f}"
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Gaussian Mixture detection failed: {e}")
            return []
    
    def _detect_kmeans_anomalies(self, feature_matrix: np.ndarray, feature_names: List[str], 
                                review_ids: List[str], reviews: List[Dict[str, Any]]) -> List[ClusteringAnomaly]:
        """Detect anomalies using K-means distance"""
        try:
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(
                n_clusters=min(self.config.kmeans_n_clusters, len(feature_matrix) // 10),
                random_state=42,
                n_init=10
            )
            
            cluster_labels = kmeans.fit_predict(feature_matrix)
            distances = kmeans.transform(feature_matrix)
            
            anomalies = []
            for i, (label, cluster_distances) in enumerate(zip(cluster_labels, distances)):
                min_distance = cluster_distances.min()
                
                if min_distance > self.config.kmeans_distance_threshold:
                    app_id = reviews[i].get('app_id', '')
                    
                    anomaly = ClusteringAnomaly(
                        review_id=review_ids[i],
                        app_id=app_id,
                        method=ClusteringMethod.KMEANS_DISTANCE,
                        score=min_distance,
                        is_anomaly=True,
                        cluster_id=int(label),
                        distance_to_cluster=min_distance,
                        explanation=f"K-means distance {min_distance:.3f} exceeds threshold {self.config.kmeans_distance_threshold}"
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"K-means detection failed: {e}")
            return []
    
    def _deduplicate_anomalies(self, anomalies: List[ClusteringAnomaly]) -> List[ClusteringAnomaly]:
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
                "dbscan_eps": self.config.dbscan_eps,
                "isolation_contamination": self.config.isolation_contamination,
                "lof_n_neighbors": self.config.lof_n_neighbors
            },
            "sklearn_available": self._sklearn_available,
            "feature_cache_size": len(self._feature_cache)
        }