"""
Smart anomaly detection pipeline using cluster dynamics.

This is a complete redesign of the anomaly detection approach:
- Focus on cluster dynamics instead of rigid rules
- Week-over-week analysis for trend detection  
- Operational vs Product issue separation
- Adaptive learning from historical patterns
"""

import asyncio
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd

from pavel.core.logger import get_logger
from .dynamic_cluster_detector import DynamicClusterDetector, DynamicAnomaly, IssueCategory

logger = get_logger(__name__)


@dataclass
class WeeklyAnalysis:
    """Analysis results for a specific week"""
    week_start: datetime
    week_end: datetime
    total_reviews: int
    clusters_found: int
    
    # Cluster distributions
    cluster_sizes: Dict[int, int]
    cluster_categories: Dict[int, IssueCategory]
    cluster_ratings: Dict[int, float]
    
    # Detected issues
    anomalies: List[DynamicAnomaly]
    operational_issues: List[DynamicAnomaly]
    product_issues: List[DynamicAnomaly]
    
    # Trends
    week_over_week_changes: Dict[str, float]
    emerging_topics: List[str]
    resolved_topics: List[str]


@dataclass 
class SmartDetectionResult:
    """Results from smart anomaly detection"""
    # Current week analysis
    current_week: WeeklyAnalysis
    
    # Historical context
    previous_weeks: List[WeeklyAnalysis]
    
    # Key insights
    critical_changes: List[DynamicAnomaly]
    operational_alerts: List[DynamicAnomaly]
    product_bugs: List[DynamicAnomaly]
    
    # Trends
    cluster_trends: Dict[int, Dict[str, Any]]
    overall_health_score: float
    
    # Recommendations
    immediate_actions: List[str]
    monitoring_suggestions: List[str]
    
    # Performance metrics
    processing_time_ms: float
    embeddings_cached: bool


class SmartDetectionPipeline:
    """
    Smart anomaly detection pipeline with cluster dynamics.
    
    Key improvements over rule-based approach:
    1. Learns what's normal from data, not predefined rules
    2. Tracks cluster evolution week-over-week
    3. Separates operational vs product issues naturally
    4. Adapts thresholds based on historical variance
    """
    
    def __init__(self,
                 embedding_pipeline=None,
                 history_weeks: int = 4,
                 min_reviews_for_analysis: int = 100):
        """
        Args:
            embedding_pipeline: Stage 4 embedding pipeline
            history_weeks: Number of weeks to use for baseline
            min_reviews_for_analysis: Minimum reviews needed
        """
        self.embedding_pipeline = embedding_pipeline
        self.history_weeks = history_weeks
        self.min_reviews_for_analysis = min_reviews_for_analysis
        
        # Dynamic cluster detector
        self.cluster_detector = DynamicClusterDetector(
            n_clusters_range=(5, 20),
            min_cluster_size=10,
            significance_threshold=2.0,
            history_weeks=history_weeks
        )
        
        # Historical data storage
        self.weekly_analyses = []
        self.cluster_evolution = defaultdict(list)
        
    async def analyze_reviews(self,
                             app_id: str,
                             reviews: List[Dict[str, Any]],
                             end_date: Optional[datetime] = None) -> SmartDetectionResult:
        """
        Perform smart anomaly detection on reviews.
        
        Args:
            app_id: Application identifier
            reviews: All available reviews
            end_date: End date for analysis period
            
        Returns:
            Smart detection results with insights and trends
        """
        start_time = datetime.now()
        
        if end_date is None:
            end_date = datetime.now()
            
        logger.info(f"Starting smart analysis for {app_id} with {len(reviews)} reviews")
        
        # 1. Organize reviews by week
        weekly_reviews = self._organize_by_week(reviews, end_date)
        
        # 2. Generate embeddings if needed
        embeddings_cached = False
        if self.embedding_pipeline:
            embeddings_map = await self._generate_embeddings(reviews)
            embeddings_cached = True
        else:
            embeddings_map = self._simple_embeddings(reviews)
        
        # 3. Analyze each week
        weekly_analyses = []
        for week_start, week_reviews in weekly_reviews.items():
            if len(week_reviews) < self.min_reviews_for_analysis:
                logger.info(f"Skipping week {week_start}: only {len(week_reviews)} reviews")
                continue
                
            analysis = await self._analyze_week(
                week_start,
                week_reviews,
                embeddings_map,
                app_id
            )
            weekly_analyses.append(analysis)
        
        # 4. Get current week analysis
        if not weekly_analyses:
            raise ValueError("No weeks had enough data for analysis")
            
        current_week = weekly_analyses[-1]
        previous_weeks = weekly_analyses[:-1] if len(weekly_analyses) > 1 else []
        
        # 5. Identify critical changes
        critical_changes = self._identify_critical_changes(current_week, previous_weeks)
        
        # 6. Separate operational vs product issues
        operational_alerts = [a for a in current_week.anomalies 
                            if a.category == IssueCategory.OPERATIONAL]
        product_bugs = [a for a in current_week.anomalies 
                       if a.category == IssueCategory.PRODUCT]
        
        # 7. Get cluster trends
        cluster_trends = self.cluster_detector.get_cluster_trends()
        
        # 8. Calculate health score
        health_score = self._calculate_health_score(current_week, previous_weeks)
        
        # 9. Generate recommendations
        immediate_actions = self._generate_immediate_actions(
            critical_changes, operational_alerts, product_bugs
        )
        monitoring_suggestions = self._generate_monitoring_suggestions(
            cluster_trends, current_week
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return SmartDetectionResult(
            current_week=current_week,
            previous_weeks=previous_weeks,
            critical_changes=critical_changes,
            operational_alerts=operational_alerts,
            product_bugs=product_bugs,
            cluster_trends=cluster_trends,
            overall_health_score=health_score,
            immediate_actions=immediate_actions,
            monitoring_suggestions=monitoring_suggestions,
            processing_time_ms=processing_time,
            embeddings_cached=embeddings_cached
        )
    
    def _organize_by_week(self, 
                         reviews: List[Dict[str, Any]], 
                         end_date: datetime) -> Dict[datetime, List[Dict[str, Any]]]:
        """
        Organize reviews into weekly buckets.
        """
        weekly_reviews = defaultdict(list)
        
        # Calculate week boundaries
        for i in range(self.history_weeks + 1):
            week_end = end_date - timedelta(weeks=i)
            week_start = week_end - timedelta(weeks=1)
            
            for review in reviews:
                review_date = review.get('created_at')
                if isinstance(review_date, str):
                    review_date = datetime.fromisoformat(review_date.replace('Z', '+00:00'))
                
                if week_start <= review_date < week_end:
                    weekly_reviews[week_start].append(review)
        
        return dict(sorted(weekly_reviews.items()))
    
    async def _generate_embeddings(self, 
                                  reviews: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings using Stage 4 pipeline.
        """
        embeddings_map = {}
        
        texts = [r.get('content', '') for r in reviews]
        review_ids = [r.get('review_id', f'review_{i}') for i, r in enumerate(reviews)]
        
        # Generate in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_ids = review_ids[i:i+batch_size]
            
            if self.embedding_pipeline:
                # Use the embedding generator directly
                batch_embeddings = await self.embedding_pipeline.embedding_generator.generate_batch_async(batch_texts)
                for rid, emb in zip(batch_ids, batch_embeddings):
                    embeddings_map[rid] = emb.embedding
            
        return embeddings_map
    
    def _simple_embeddings(self, reviews: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Generate simple embeddings when Stage 4 not available.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        texts = [r.get('content', '') for r in reviews]
        review_ids = [r.get('review_id', f'review_{i}') for i, r in enumerate(reviews)]
        
        # Use TF-IDF as fallback
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        embeddings = vectorizer.fit_transform(texts).toarray()
        
        embeddings_map = {}
        for rid, emb in zip(review_ids, embeddings):
            embeddings_map[rid] = emb
            
        return embeddings_map
    
    async def _analyze_week(self,
                           week_start: datetime,
                           reviews: List[Dict[str, Any]],
                           embeddings_map: Dict[str, np.ndarray],
                           app_id: str) -> WeeklyAnalysis:
        """
        Analyze a single week of reviews.
        """
        # Get embeddings for this week's reviews
        review_ids = [r.get('review_id', f'review_{i}') for i, r in enumerate(reviews)]
        embeddings = np.array([embeddings_map.get(rid, np.zeros(100)) for rid in review_ids])
        
        # Detect anomalies using cluster dynamics
        anomalies = self.cluster_detector.detect_anomalies(
            reviews=reviews,
            embeddings=embeddings,
            current_week=week_start
        )
        
        # Separate by category
        operational = [a for a in anomalies if a.category == IssueCategory.OPERATIONAL]
        product = [a for a in anomalies if a.category == IssueCategory.PRODUCT]
        
        # Get cluster info
        clusters = self.cluster_detector.cluster_baselines
        cluster_sizes = {cid: c['size'] for cid, c in clusters.items()}
        cluster_categories = {cid: c.get('category', IssueCategory.UNKNOWN) 
                             for cid, c in clusters.items()}
        cluster_ratings = {cid: c.get('mean_rating', 3.0) for cid, c in clusters.items()}
        
        # Calculate week-over-week changes
        wow_changes = self._calculate_wow_changes(week_start)
        
        # Identify emerging and resolved topics
        emerging = self._find_emerging_topics(anomalies)
        resolved = self._find_resolved_topics(week_start)
        
        return WeeklyAnalysis(
            week_start=week_start,
            week_end=week_start + timedelta(weeks=1),
            total_reviews=len(reviews),
            clusters_found=len(clusters),
            cluster_sizes=cluster_sizes,
            cluster_categories=cluster_categories,
            cluster_ratings=cluster_ratings,
            anomalies=anomalies,
            operational_issues=operational,
            product_issues=product,
            week_over_week_changes=wow_changes,
            emerging_topics=emerging,
            resolved_topics=resolved
        )
    
    def _calculate_wow_changes(self, week_start: datetime) -> Dict[str, float]:
        """
        Calculate week-over-week changes.
        """
        changes = {}
        
        if len(self.weekly_analyses) > 0:
            prev_week = self.weekly_analyses[-1]
            
            # Review volume change
            current_clusters = self.cluster_detector.cluster_baselines
            total_current = sum(c['size'] for c in current_clusters.values())
            total_prev = prev_week.total_reviews
            
            if total_prev > 0:
                changes['volume_change_pct'] = (total_current - total_prev) / total_prev * 100
            
            # Average rating change
            current_ratings = [c.get('mean_rating', 3.0) for c in current_clusters.values()]
            if current_ratings:
                changes['rating_change'] = np.mean(current_ratings) - np.mean(list(prev_week.cluster_ratings.values()))
            
            # Anomaly count change
            changes['anomaly_change'] = len(current_clusters) - len(prev_week.anomalies)
        
        return changes
    
    def _find_emerging_topics(self, anomalies: List[DynamicAnomaly]) -> List[str]:
        """
        Find emerging topics from anomalies.
        """
        emerging = []
        
        for anomaly in anomalies:
            if anomaly.anomaly_type == "emerging_cluster":
                keywords = anomaly.cluster_profile.dominant_keywords[:3]
                emerging.append(f"{anomaly.category.value}: {', '.join(keywords)}")
        
        return emerging
    
    def _find_resolved_topics(self, week_start: datetime) -> List[str]:
        """
        Find topics that were resolved (clusters that disappeared).
        """
        # This would track clusters that were present in previous weeks
        # but are no longer significant
        return []
    
    def _identify_critical_changes(self,
                                  current: WeeklyAnalysis,
                                  previous: List[WeeklyAnalysis]) -> List[DynamicAnomaly]:
        """
        Identify the most critical changes requiring attention.
        """
        critical = []
        
        # High severity anomalies
        for anomaly in current.anomalies:
            if anomaly.severity > 7.0:  # High severity threshold
                critical.append(anomaly)
        
        # Large operational issues
        for anomaly in current.operational_issues:
            if anomaly.change_magnitude > 100:  # 100% change
                critical.append(anomaly)
        
        # New product bugs with high volume
        for anomaly in current.product_issues:
            if anomaly.anomaly_type == "emerging_cluster" and \
               anomaly.cluster_profile.size > 20:
                critical.append(anomaly)
        
        # Remove duplicates
        seen = set()
        unique_critical = []
        for anomaly in critical:
            key = (anomaly.cluster_id, anomaly.anomaly_type)
            if key not in seen:
                seen.add(key)
                unique_critical.append(anomaly)
        
        return unique_critical
    
    def _calculate_health_score(self,
                               current: WeeklyAnalysis,
                               previous: List[WeeklyAnalysis]) -> float:
        """
        Calculate overall health score (0-100).
        """
        score = 100.0
        
        # Penalize for anomalies
        score -= len(current.anomalies) * 2
        score -= len(current.operational_issues) * 5  # Operational issues are worse
        score -= len(current.product_issues) * 3
        
        # Penalize for negative trends
        if current.week_over_week_changes.get('rating_change', 0) < -0.5:
            score -= 10
        
        if current.week_over_week_changes.get('volume_change_pct', 0) > 200:
            score -= 15  # Suspicious volume spike
        
        # Bonus for improvements
        if current.week_over_week_changes.get('rating_change', 0) > 0.5:
            score += 5
        
        if len(current.resolved_topics) > 0:
            score += len(current.resolved_topics) * 2
        
        return max(0, min(100, score))
    
    def _generate_immediate_actions(self,
                                   critical: List[DynamicAnomaly],
                                   operational: List[DynamicAnomaly], 
                                   product: List[DynamicAnomaly]) -> List[str]:
        """
        Generate immediate action recommendations.
        """
        actions = []
        
        if operational:
            actions.append(f"🚨 OPERATIONAL: {len(operational)} issues detected - check service health")
            
            for op in operational[:3]:  # Top 3
                actions.append(f"  → {op.suggested_action}")
        
        if product:
            actions.append(f"🐛 PRODUCT: {len(product)} potential bugs - review recent deployments")
            
            for pr in product[:3]:  # Top 3
                actions.append(f"  → {pr.suggested_action}")
        
        if critical and not operational and not product:
            actions.append("⚠️ CRITICAL: Unusual patterns detected - investigate manually")
            
            for cr in critical[:3]:
                actions.append(f"  → Cluster {cr.cluster_id}: {cr.explanation}")
        
        if not actions:
            actions.append("✅ No immediate actions required - system healthy")
        
        return actions
    
    def _generate_monitoring_suggestions(self,
                                        trends: Dict[int, Dict[str, Any]],
                                        current: WeeklyAnalysis) -> List[str]:
        """
        Generate monitoring suggestions.
        """
        suggestions = []
        
        # Growing clusters
        growing = [cid for cid, trend in trends.items() 
                  if trend.get('size_trend') == 'growing']
        if growing:
            suggestions.append(f"📈 Monitor growing clusters: {growing[:3]}")
        
        # Declining ratings
        declining = [cid for cid, trend in trends.items()
                    if trend.get('rating_trend') == 'declining']
        if declining:
            suggestions.append(f"📉 Watch declining satisfaction in clusters: {declining[:3]}")
        
        # Emerging topics
        if current.emerging_topics:
            suggestions.append(f"🆕 Track emerging topics: {', '.join(current.emerging_topics[:3])}")
        
        # Stability monitoring
        if not suggestions:
            suggestions.append("📊 Continue regular monitoring - no special focus needed")
        
        return suggestions