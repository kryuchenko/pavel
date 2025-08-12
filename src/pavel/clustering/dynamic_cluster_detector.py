"""
Dynamic cluster-based anomaly detection for Google Play reviews.

Instead of rigid rule-based detection, uses semantic clustering
and week-over-week dynamics to identify anomalies.

Key principles:
1. Semantic clustering to identify review themes
2. Week-over-week analysis of cluster dynamics
3. Separation of operational vs product issues
4. Anomalies = significant changes in cluster patterns
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

from pavel.core.logger import get_logger

logger = get_logger(__name__)


class IssueCategory(Enum):
    """Issue categorization based on nature"""
    OPERATIONAL = "operational"      # Service issues, payments, auth
    PRODUCT = "product"              # App bugs, crashes, features
    CONTENT = "content"              # UX/UI, localization, content
    PERFORMANCE = "performance"      # Speed, battery, memory
    SENTIMENT = "sentiment"          # General satisfaction changes
    UNKNOWN = "unknown"


@dataclass
class ClusterProfile:
    """Profile of a semantic cluster"""
    cluster_id: int
    size: int
    centroid: np.ndarray
    mean_rating: float
    std_rating: float
    dominant_keywords: List[str]
    category: IssueCategory
    time_window: datetime
    
    # Dynamic metrics
    size_change_pct: float = 0.0
    rating_shift: float = 0.0
    centroid_drift: float = 0.0
    is_emerging: bool = False
    is_declining: bool = False


@dataclass
class DynamicAnomaly:
    """Anomaly detected through cluster dynamics"""
    cluster_id: int
    anomaly_type: str
    severity: float
    category: IssueCategory
    
    # Dynamics
    baseline_week: datetime
    current_week: datetime
    change_magnitude: float
    
    # Context
    affected_reviews: List[str]
    cluster_profile: ClusterProfile
    explanation: str
    
    # Recommendations
    is_actionable: bool
    suggested_action: str


class DynamicClusterDetector:
    """
    Dynamic cluster-based anomaly detector.
    
    Identifies anomalies through:
    1. Semantic clustering of reviews
    2. Week-over-week cluster dynamics
    3. Operational vs product issue classification
    4. Adaptive threshold learning
    """
    
    def __init__(self, 
                 n_clusters_range: Tuple[int, int] = (5, 20),
                 min_cluster_size: int = 10,
                 significance_threshold: float = 2.0,  # Standard deviations
                 history_weeks: int = 4):
        """
        Args:
            n_clusters_range: Min and max clusters to discover
            min_cluster_size: Minimum reviews in a cluster
            significance_threshold: Z-score for significant change
            history_weeks: Weeks of history for baseline
        """
        self.n_clusters_range = n_clusters_range
        self.min_cluster_size = min_cluster_size
        self.significance_threshold = significance_threshold
        self.history_weeks = history_weeks
        
        # Learned baselines
        self.cluster_baselines = {}
        self.historical_dynamics = defaultdict(list)
        
    def detect_anomalies(self, 
                        reviews: List[Dict[str, Any]], 
                        embeddings: np.ndarray,
                        current_week: Optional[datetime] = None) -> List[DynamicAnomaly]:
        """
        Detect anomalies through cluster dynamics.
        
        Args:
            reviews: Current week's reviews
            embeddings: Review embeddings
            current_week: Week identifier
            
        Returns:
            List of detected dynamic anomalies
        """
        if current_week is None:
            current_week = datetime.now()
            
        logger.info(f"Analyzing {len(reviews)} reviews for week {current_week}")
        
        # 1. Perform adaptive clustering
        clusters = self._adaptive_clustering(embeddings, reviews)
        
        # 2. Build cluster profiles
        profiles = self._build_cluster_profiles(clusters, embeddings, reviews, current_week)
        
        # 3. Compare with historical baselines
        anomalies = self._detect_dynamic_anomalies(profiles, current_week)
        
        # 4. Update baselines
        self._update_baselines(profiles)
        
        logger.info(f"Detected {len(anomalies)} dynamic anomalies")
        return anomalies
    
    def _adaptive_clustering(self, 
                            embeddings: np.ndarray, 
                            reviews: List[Dict[str, Any]]) -> Dict[int, List[int]]:
        """
        Perform adaptive clustering to find natural groups.
        """
        # Try HDBSCAN first for natural cluster discovery
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=5,
            metric='euclidean',
            cluster_selection_epsilon=0.0,
            cluster_selection_method='eom'
        )
        
        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_normalized = scaler.fit_transform(embeddings)
        
        # Perform clustering
        cluster_labels = clusterer.fit_predict(embeddings_normalized)
        
        # Count valid clusters (excluding noise -1)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        logger.debug(f"HDBSCAN found {n_clusters} clusters")
        
        # If HDBSCAN finds too few/many clusters, use KMeans as fallback
        if n_clusters < self.n_clusters_range[0] or n_clusters > self.n_clusters_range[1]:
            optimal_k = self._find_optimal_k(embeddings_normalized)
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_normalized)
            logger.debug(f"Using KMeans with k={optimal_k}")
        
        # Group indices by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            if label != -1:  # Exclude noise
                clusters[label].append(idx)
        
        return dict(clusters)
    
    def _find_optimal_k(self, embeddings: np.ndarray) -> int:
        """
        Find optimal number of clusters using elbow method.
        """
        min_k, max_k = self.n_clusters_range
        inertias = []
        
        for k in range(min_k, min(max_k + 1, len(embeddings) // 10)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point (simplified - just use middle of range)
        # In production, use more sophisticated elbow detection
        optimal_k = min_k + len(inertias) // 2
        
        return optimal_k
    
    def _build_cluster_profiles(self,
                               clusters: Dict[int, List[int]],
                               embeddings: np.ndarray,
                               reviews: List[Dict[str, Any]],
                               current_week: datetime) -> List[ClusterProfile]:
        """
        Build detailed profiles for each cluster.
        """
        profiles = []
        
        for cluster_id, indices in clusters.items():
            cluster_embeddings = embeddings[indices]
            cluster_reviews = [reviews[i] for i in indices]
            
            # Calculate cluster metrics
            centroid = np.mean(cluster_embeddings, axis=0)
            
            ratings = [r.get('rating', 3) for r in cluster_reviews]
            mean_rating = np.mean(ratings)
            std_rating = np.std(ratings)
            
            # Extract keywords (simplified - use most common words)
            keywords = self._extract_keywords(cluster_reviews)
            
            # Classify cluster category
            category = self._classify_cluster_category(cluster_reviews, keywords)
            
            # Check dynamics if we have history
            size_change_pct = 0.0
            rating_shift = 0.0
            centroid_drift = 0.0
            is_emerging = False
            is_declining = False
            
            if cluster_id in self.cluster_baselines:
                baseline = self.cluster_baselines[cluster_id]
                size_change_pct = (len(indices) - baseline['size']) / baseline['size'] * 100
                rating_shift = mean_rating - baseline['mean_rating']
                centroid_drift = np.linalg.norm(centroid - baseline['centroid'])
                
                # Detect emerging/declining patterns
                is_emerging = size_change_pct > 50 and len(self.historical_dynamics[cluster_id]) < 2
                is_declining = size_change_pct < -50
            
            profile = ClusterProfile(
                cluster_id=cluster_id,
                size=len(indices),
                centroid=centroid,
                mean_rating=mean_rating,
                std_rating=std_rating,
                dominant_keywords=keywords,
                category=category,
                time_window=current_week,
                size_change_pct=size_change_pct,
                rating_shift=rating_shift,
                centroid_drift=centroid_drift,
                is_emerging=is_emerging,
                is_declining=is_declining
            )
            
            profiles.append(profile)
        
        return profiles
    
    def _extract_keywords(self, reviews: List[Dict[str, Any]], top_n: int = 5) -> List[str]:
        """
        Extract dominant keywords from cluster reviews.
        """
        # Simple word frequency approach
        word_counts = Counter()
        
        for review in reviews:
            content = review.get('content', '').lower()
            words = content.split()
            
            # Filter stop words and short words
            meaningful_words = [w for w in words if len(w) > 3]
            word_counts.update(meaningful_words)
        
        # Return top keywords
        return [word for word, _ in word_counts.most_common(top_n)]
    
    def _classify_cluster_category(self, 
                                  reviews: List[Dict[str, Any]], 
                                  keywords: List[str]) -> IssueCategory:
        """
        Classify cluster into operational vs product categories.
        """
        # Keywords indicating different categories
        operational_keywords = {'payment', 'login', 'account', 'charge', 'subscription', 
                               'refund', 'billing', 'auth', 'register', 'sign'}
        product_keywords = {'crash', 'bug', 'error', 'freeze', 'broken', 'fix', 
                           'update', 'feature', 'function', 'work'}
        performance_keywords = {'slow', 'fast', 'speed', 'lag', 'battery', 'memory',
                               'performance', 'responsive', 'smooth', 'heavy'}
        content_keywords = {'interface', 'design', 'ui', 'ux', 'button', 'screen',
                           'navigation', 'menu', 'layout', 'display'}
        
        # Count category indicators
        categories_score = {
            IssueCategory.OPERATIONAL: 0,
            IssueCategory.PRODUCT: 0,
            IssueCategory.PERFORMANCE: 0,
            IssueCategory.CONTENT: 0,
            IssueCategory.SENTIMENT: 0
        }
        
        # Score based on keywords
        for keyword in keywords:
            if keyword in operational_keywords:
                categories_score[IssueCategory.OPERATIONAL] += 2
            if keyword in product_keywords:
                categories_score[IssueCategory.PRODUCT] += 2
            if keyword in performance_keywords:
                categories_score[IssueCategory.PERFORMANCE] += 2
            if keyword in content_keywords:
                categories_score[IssueCategory.CONTENT] += 2
        
        # Also check review content
        for review in reviews[:10]:  # Sample first 10
            content_lower = review.get('content', '').lower()
            
            if any(kw in content_lower for kw in operational_keywords):
                categories_score[IssueCategory.OPERATIONAL] += 1
            if any(kw in content_lower for kw in product_keywords):
                categories_score[IssueCategory.PRODUCT] += 1
            if any(kw in content_lower for kw in performance_keywords):
                categories_score[IssueCategory.PERFORMANCE] += 1
            if any(kw in content_lower for kw in content_keywords):
                categories_score[IssueCategory.CONTENT] += 1
        
        # Get category with highest score
        max_category = max(categories_score, key=categories_score.get)
        
        # If no clear category, mark as sentiment
        if categories_score[max_category] < 3:
            return IssueCategory.SENTIMENT
        
        return max_category
    
    def _detect_dynamic_anomalies(self, 
                                 profiles: List[ClusterProfile],
                                 current_week: datetime) -> List[DynamicAnomaly]:
        """
        Detect anomalies by comparing current profiles with baselines.
        """
        anomalies = []
        
        for profile in profiles:
            cluster_id = profile.cluster_id
            
            # Skip if no baseline yet
            if cluster_id not in self.cluster_baselines:
                continue
            
            baseline = self.cluster_baselines[cluster_id]
            history = self.historical_dynamics[cluster_id]
            
            # Detect various types of anomalies
            
            # 1. Volume spike/drop
            if len(history) >= 2:
                sizes = [h['size'] for h in history]
                mean_size = np.mean(sizes)
                std_size = np.std(sizes) if np.std(sizes) > 0 else 1
                z_score = (profile.size - mean_size) / std_size
                
                if abs(z_score) > self.significance_threshold:
                    anomaly_type = "volume_spike" if z_score > 0 else "volume_drop"
                    
                    anomaly = DynamicAnomaly(
                        cluster_id=cluster_id,
                        anomaly_type=anomaly_type,
                        severity=min(abs(z_score), 10.0),  # Cap at 10
                        category=profile.category,
                        baseline_week=baseline['week'],
                        current_week=current_week,
                        change_magnitude=profile.size_change_pct,
                        affected_reviews=[],  # Would fill with actual review IDs
                        cluster_profile=profile,
                        explanation=f"Cluster size changed by {profile.size_change_pct:.1f}% (z-score: {z_score:.2f})",
                        is_actionable=True,
                        suggested_action=self._suggest_action(anomaly_type, profile.category)
                    )
                    anomalies.append(anomaly)
            
            # 2. Rating shift
            if abs(profile.rating_shift) > 1.0:
                anomaly = DynamicAnomaly(
                    cluster_id=cluster_id,
                    anomaly_type="rating_shift",
                    severity=abs(profile.rating_shift) * 2,
                    category=profile.category,
                    baseline_week=baseline['week'],
                    current_week=current_week,
                    change_magnitude=profile.rating_shift,
                    affected_reviews=[],
                    cluster_profile=profile,
                    explanation=f"Average rating shifted by {profile.rating_shift:.1f} stars",
                    is_actionable=True,
                    suggested_action=self._suggest_action("rating_shift", profile.category)
                )
                anomalies.append(anomaly)
            
            # 3. Semantic drift
            if profile.centroid_drift > 0.5:  # Threshold for significant drift
                anomaly = DynamicAnomaly(
                    cluster_id=cluster_id,
                    anomaly_type="semantic_drift",
                    severity=profile.centroid_drift * 5,
                    category=profile.category,
                    baseline_week=baseline['week'],
                    current_week=current_week,
                    change_magnitude=profile.centroid_drift,
                    affected_reviews=[],
                    cluster_profile=profile,
                    explanation=f"Cluster semantics drifted significantly (distance: {profile.centroid_drift:.2f})",
                    is_actionable=True,
                    suggested_action="Review cluster content for emerging issues or topic shifts"
                )
                anomalies.append(anomaly)
            
            # 4. Emerging cluster
            if profile.is_emerging:
                anomaly = DynamicAnomaly(
                    cluster_id=cluster_id,
                    anomaly_type="emerging_cluster",
                    severity=5.0,  # Medium-high severity for new patterns
                    category=profile.category,
                    baseline_week=current_week,
                    current_week=current_week,
                    change_magnitude=100.0,  # New cluster
                    affected_reviews=[],
                    cluster_profile=profile,
                    explanation=f"New cluster emerged with {profile.size} reviews about: {', '.join(profile.dominant_keywords[:3])}",
                    is_actionable=True,
                    suggested_action=self._suggest_action("emerging_cluster", profile.category)
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _suggest_action(self, anomaly_type: str, category: IssueCategory) -> str:
        """
        Generate actionable suggestions based on anomaly type and category.
        """
        suggestions = {
            ("volume_spike", IssueCategory.OPERATIONAL): 
                "⚠️ Investigate operational issues - possible service disruption or payment problems",
            ("volume_spike", IssueCategory.PRODUCT): 
                "🐛 Check for new bugs or crashes - possible regression in recent release",
            ("volume_drop", IssueCategory.OPERATIONAL): 
                "✅ Operational issues may have been resolved - verify fix deployment",
            ("rating_shift", IssueCategory.PRODUCT): 
                "📉 Product quality impact detected - review recent changes and rollback if needed",
            ("emerging_cluster", IssueCategory.OPERATIONAL): 
                "🚨 New operational issue pattern - escalate to operations team immediately",
            ("emerging_cluster", IssueCategory.PRODUCT): 
                "🔍 New product issue emerging - investigate and prioritize fix"
        }
        
        key = (anomaly_type, category)
        if key in suggestions:
            return suggestions[key]
        
        # Default suggestion
        if category == IssueCategory.OPERATIONAL:
            return "Review operational metrics and logs for issues"
        elif category == IssueCategory.PRODUCT:
            return "Investigate product issues and recent code changes"
        else:
            return "Monitor trend and gather more data"
    
    def _update_baselines(self, profiles: List[ClusterProfile]):
        """
        Update historical baselines with current profiles.
        """
        for profile in profiles:
            cluster_id = profile.cluster_id
            
            # Store current as baseline
            baseline = {
                'week': profile.time_window,
                'size': profile.size,
                'mean_rating': profile.mean_rating,
                'centroid': profile.centroid,
                'category': profile.category
            }
            
            # Update baseline
            self.cluster_baselines[cluster_id] = baseline
            
            # Add to history
            self.historical_dynamics[cluster_id].append(baseline)
            
            # Keep only recent history
            if len(self.historical_dynamics[cluster_id]) > self.history_weeks:
                self.historical_dynamics[cluster_id].pop(0)
    
    def get_cluster_trends(self) -> Dict[int, Dict[str, Any]]:
        """
        Get trend analysis for all tracked clusters.
        """
        trends = {}
        
        for cluster_id, history in self.historical_dynamics.items():
            if len(history) < 2:
                continue
            
            sizes = [h['size'] for h in history]
            ratings = [h['mean_rating'] for h in history]
            
            # Calculate trends
            size_trend = "growing" if sizes[-1] > sizes[0] else "shrinking"
            rating_trend = "improving" if ratings[-1] > ratings[0] else "declining"
            
            trends[cluster_id] = {
                'size_trend': size_trend,
                'rating_trend': rating_trend,
                'current_size': sizes[-1],
                'size_change_pct': (sizes[-1] - sizes[0]) / sizes[0] * 100 if sizes[0] > 0 else 0,
                'current_rating': ratings[-1],
                'rating_change': ratings[-1] - ratings[0],
                'history_weeks': len(history)
            }
        
        return trends