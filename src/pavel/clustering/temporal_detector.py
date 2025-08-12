"""
Temporal anomaly detection for Google Play reviews.

Detects time-based anomalies such as sudden spikes, drops, 
seasonal changes, and trend deviations in review patterns.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import math

from pavel.core.logger import get_logger

logger = get_logger(__name__)

class TemporalMethod(Enum):
    """Temporal anomaly detection methods"""
    VOLUME_SPIKE = "volume_spike"
    VOLUME_DROP = "volume_drop"
    RATING_SHIFT = "rating_shift"
    SENTIMENT_CHANGE = "sentiment_change"
    SEASONAL_DEVIATION = "seasonal_deviation"
    TREND_BREAK = "trend_break"
    PERIODICITY_ANOMALY = "periodicity_anomaly"
    TIME_CLUSTERING = "time_clustering"

@dataclass
class TemporalConfig:
    """Configuration for temporal anomaly detection"""
    # Time window configurations
    spike_window_hours: int = 24
    trend_window_days: int = 7
    seasonal_window_days: int = 30
    
    # Volume thresholds
    volume_spike_multiplier: float = 3.0  # 3x normal volume
    volume_drop_threshold: float = 0.3  # 30% of normal volume
    
    # Rating shift thresholds
    rating_shift_threshold: float = 1.0  # 1 star difference
    rating_significance_threshold: int = 10  # Minimum reviews for significance
    
    # Sentiment change thresholds
    sentiment_shift_threshold: float = 0.3  # 30% sentiment change
    
    # Trend analysis
    trend_significance_threshold: float = 0.05  # p-value for trend significance
    trend_break_threshold: float = 0.5  # Correlation threshold
    
    # Seasonal analysis
    seasonal_deviation_threshold: float = 2.0  # Standard deviations
    
    # Periodicity
    periodicity_min_cycles: int = 3
    periodicity_confidence_threshold: float = 0.7
    
    # Methods to apply
    methods: List[TemporalMethod] = None
    
    # Minimum sample size
    min_sample_size: int = 20
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = [
                TemporalMethod.VOLUME_SPIKE,
                TemporalMethod.VOLUME_DROP,
                TemporalMethod.RATING_SHIFT,
                TemporalMethod.TREND_BREAK
            ]

@dataclass
class TemporalAnomaly:
    """Temporal anomaly detection result"""
    anomaly_id: str
    app_id: str
    method: TemporalMethod
    score: float
    is_anomaly: bool
    time_window_start: datetime
    time_window_end: datetime
    affected_reviews: List[str]
    baseline_value: Optional[float] = None
    anomaly_value: Optional[float] = None
    statistical_significance: Optional[float] = None
    explanation: Optional[str] = None

class TemporalAnomalyDetector:
    """
    Temporal anomaly detector for Google Play reviews.
    
    Detects time-based patterns and anomalies:
    - Volume spikes/drops in review frequency
    - Sudden rating shifts over time
    - Sentiment changes in review content
    - Seasonal deviations from expected patterns
    - Trend breaks and unexpected changes
    - Periodicity anomalies
    - Time-based clustering outliers
    
    Analyzes reviews over various time windows to identify
    unusual temporal behaviors that might indicate significant events.
    """
    
    def __init__(self, config: Optional[TemporalConfig] = None):
        self.config = config or TemporalConfig()
    
    def detect_anomalies(self, reviews: List[Dict[str, Any]]) -> List[TemporalAnomaly]:
        """
        Detect temporal anomalies in reviews.
        
        Args:
            reviews: List of review dictionaries with timestamps
            
        Returns:
            List of detected temporal anomalies
        """
        if len(reviews) < self.config.min_sample_size:
            logger.warning(f"Sample size {len(reviews)} below minimum {self.config.min_sample_size}")
            return []
        
        logger.info(f"Analyzing {len(reviews)} reviews for temporal anomalies")
        
        # Prepare temporal data
        temporal_df = self._prepare_temporal_data(reviews)
        
        if temporal_df is None or len(temporal_df) == 0:
            logger.error("No temporal data available for analysis")
            return []
        
        logger.debug(f"Prepared temporal dataset with {len(temporal_df)} time points")
        
        anomalies = []
        
        # Apply each configured method
        for method in self.config.methods:
            try:
                method_anomalies = self._apply_temporal_method(method, temporal_df, reviews)
                anomalies.extend(method_anomalies)
                logger.debug(f"Method {method.value} found {len(method_anomalies)} anomalies")
            except Exception as e:
                logger.error(f"Temporal method {method.value} failed: {e}")
                continue
        
        # Remove duplicates and sort by significance
        unique_anomalies = self._deduplicate_anomalies(anomalies)
        
        logger.info(f"Found {len(unique_anomalies)} unique temporal anomalies")
        return unique_anomalies
    
    def _prepare_temporal_data(self, reviews: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """Prepare temporal DataFrame from reviews"""
        temporal_data = []
        
        for review in reviews:
            timestamp = review.get('created_at')
            if not timestamp:
                continue
            
            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                elif isinstance(timestamp, datetime):
                    dt = timestamp
                else:
                    continue
                
                # Extract temporal features
                row = {
                    'timestamp': dt,
                    'review_id': review.get('review_id', ''),
                    'app_id': review.get('app_id', ''),
                    'rating': review.get('rating', 0),
                    'content_length': len(review.get('content', '')),
                    'hour': dt.hour,
                    'day_of_week': dt.weekday(),
                    'day_of_month': dt.day,
                    'month': dt.month,
                    'year': dt.year,
                    'is_weekend': dt.weekday() >= 5,
                    'quarter': (dt.month - 1) // 3 + 1
                }
                
                # Simple sentiment analysis
                content = review.get('content', '').lower()
                positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'awesome']
                negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'worst', 'sucks']
                
                pos_count = sum(1 for word in positive_words if word in content)
                neg_count = sum(1 for word in negative_words if word in content)
                
                if pos_count + neg_count > 0:
                    row['sentiment_score'] = (pos_count - neg_count) / (pos_count + neg_count)
                else:
                    row['sentiment_score'] = 0.0
                
                temporal_data.append(row)
                
            except Exception as e:
                logger.debug(f"Failed to parse timestamp {timestamp}: {e}")
                continue
        
        if not temporal_data:
            return None
        
        df = pd.DataFrame(temporal_data)
        df = df.sort_values('timestamp')
        
        return df
    
    def _apply_temporal_method(self, method: TemporalMethod, temporal_df: pd.DataFrame, 
                              reviews: List[Dict[str, Any]]) -> List[TemporalAnomaly]:
        """Apply specific temporal method"""
        if method == TemporalMethod.VOLUME_SPIKE:
            return self._detect_volume_spikes(temporal_df)
        elif method == TemporalMethod.VOLUME_DROP:
            return self._detect_volume_drops(temporal_df)
        elif method == TemporalMethod.RATING_SHIFT:
            return self._detect_rating_shifts(temporal_df)
        elif method == TemporalMethod.SENTIMENT_CHANGE:
            return self._detect_sentiment_changes(temporal_df)
        elif method == TemporalMethod.SEASONAL_DEVIATION:
            return self._detect_seasonal_deviations(temporal_df)
        elif method == TemporalMethod.TREND_BREAK:
            return self._detect_trend_breaks(temporal_df)
        elif method == TemporalMethod.PERIODICITY_ANOMALY:
            return self._detect_periodicity_anomalies(temporal_df)
        elif method == TemporalMethod.TIME_CLUSTERING:
            return self._detect_time_clustering_anomalies(temporal_df)
        
        return []
    
    def _detect_volume_spikes(self, temporal_df: pd.DataFrame) -> List[TemporalAnomaly]:
        """Detect volume spikes in review frequency"""
        anomalies = []
        
        # Group by time windows
        temporal_df['time_window'] = temporal_df['timestamp'].dt.floor(f'{self.config.spike_window_hours}h')
        volume_by_window = temporal_df.groupby('time_window').size()
        
        if len(volume_by_window) < 3:
            return anomalies
        
        # Calculate baseline (median of recent windows)
        baseline_volume = volume_by_window.median()
        volume_std = volume_by_window.std()
        
        # Detect spikes
        for time_window, volume in volume_by_window.items():
            spike_threshold = baseline_volume * self.config.volume_spike_multiplier
            
            if volume > spike_threshold and volume > baseline_volume + 2 * volume_std:
                # Get affected reviews
                affected_reviews = temporal_df[
                    temporal_df['time_window'] == time_window
                ]['review_id'].tolist()
                
                app_ids = temporal_df[temporal_df['time_window'] == time_window]['app_id'].unique()
                primary_app_id = app_ids[0] if len(app_ids) > 0 else ''
                
                anomaly = TemporalAnomaly(
                    anomaly_id=f"volume_spike_{time_window.strftime('%Y%m%d_%H%M')}",
                    app_id=primary_app_id,
                    method=TemporalMethod.VOLUME_SPIKE,
                    score=volume / baseline_volume if baseline_volume > 0 else volume,
                    is_anomaly=True,
                    time_window_start=time_window,
                    time_window_end=time_window + timedelta(hours=self.config.spike_window_hours),
                    affected_reviews=affected_reviews,
                    baseline_value=baseline_volume,
                    anomaly_value=volume,
                    explanation=f"Volume spike: {volume} reviews (baseline: {baseline_volume:.1f})"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_volume_drops(self, temporal_df: pd.DataFrame) -> List[TemporalAnomaly]:
        """Detect volume drops in review frequency"""
        anomalies = []
        
        # Group by time windows
        temporal_df['time_window'] = temporal_df['timestamp'].dt.floor(f'{self.config.spike_window_hours}h')
        volume_by_window = temporal_df.groupby('time_window').size()
        
        if len(volume_by_window) < 3:
            return anomalies
        
        # Calculate baseline
        baseline_volume = volume_by_window.median()
        
        # Detect drops
        for time_window, volume in volume_by_window.items():
            drop_threshold = baseline_volume * self.config.volume_drop_threshold
            
            if volume < drop_threshold and baseline_volume > 0:
                affected_reviews = temporal_df[
                    temporal_df['time_window'] == time_window
                ]['review_id'].tolist()
                
                app_ids = temporal_df[temporal_df['time_window'] == time_window]['app_id'].unique()
                primary_app_id = app_ids[0] if len(app_ids) > 0 else ''
                
                anomaly = TemporalAnomaly(
                    anomaly_id=f"volume_drop_{time_window.strftime('%Y%m%d_%H%M')}",
                    app_id=primary_app_id,
                    method=TemporalMethod.VOLUME_DROP,
                    score=baseline_volume / volume if volume > 0 else baseline_volume,
                    is_anomaly=True,
                    time_window_start=time_window,
                    time_window_end=time_window + timedelta(hours=self.config.spike_window_hours),
                    affected_reviews=affected_reviews,
                    baseline_value=baseline_volume,
                    anomaly_value=volume,
                    explanation=f"Volume drop: {volume} reviews (baseline: {baseline_volume:.1f})"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_rating_shifts(self, temporal_df: pd.DataFrame) -> List[TemporalAnomaly]:
        """Detect sudden rating shifts"""
        anomalies = []
        
        # Group by daily windows
        temporal_df['date'] = temporal_df['timestamp'].dt.date
        daily_stats = temporal_df.groupby('date').agg({
            'rating': ['mean', 'count'],
            'review_id': 'first',
            'app_id': 'first'
        }).reset_index()
        
        daily_stats.columns = ['date', 'avg_rating', 'review_count', 'sample_review_id', 'app_id']
        
        # Filter days with significant review counts
        significant_days = daily_stats[
            daily_stats['review_count'] >= self.config.rating_significance_threshold
        ]
        
        if len(significant_days) < 2:
            return anomalies
        
        # Calculate rolling baseline
        significant_days['baseline_rating'] = significant_days['avg_rating'].rolling(
            window=min(7, len(significant_days)), 
            min_periods=1
        ).mean()
        
        # Detect shifts
        for idx, row in significant_days.iterrows():
            if pd.isna(row['baseline_rating']):
                continue
                
            rating_shift = abs(row['avg_rating'] - row['baseline_rating'])
            
            if rating_shift > self.config.rating_shift_threshold:
                # Get reviews from that day
                day_reviews = temporal_df[temporal_df['date'] == row['date']]
                affected_reviews = day_reviews['review_id'].tolist()
                
                anomaly = TemporalAnomaly(
                    anomaly_id=f"rating_shift_{row['date']}",
                    app_id=row['app_id'],
                    method=TemporalMethod.RATING_SHIFT,
                    score=rating_shift,
                    is_anomaly=True,
                    time_window_start=datetime.combine(row['date'], datetime.min.time()),
                    time_window_end=datetime.combine(row['date'], datetime.max.time()),
                    affected_reviews=affected_reviews,
                    baseline_value=row['baseline_rating'],
                    anomaly_value=row['avg_rating'],
                    explanation=f"Rating shift: {row['avg_rating']:.1f} vs baseline {row['baseline_rating']:.1f}"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_sentiment_changes(self, temporal_df: pd.DataFrame) -> List[TemporalAnomaly]:
        """Detect sudden sentiment changes"""
        anomalies = []
        
        # Group by daily windows
        temporal_df['date'] = temporal_df['timestamp'].dt.date
        daily_sentiment = temporal_df.groupby('date').agg({
            'sentiment_score': ['mean', 'count'],
            'review_id': lambda x: list(x),
            'app_id': 'first'
        }).reset_index()
        
        daily_sentiment.columns = ['date', 'avg_sentiment', 'review_count', 'review_ids', 'app_id']
        
        # Filter days with sufficient reviews
        significant_days = daily_sentiment[daily_sentiment['review_count'] >= 5]
        
        if len(significant_days) < 2:
            return anomalies
        
        # Calculate rolling baseline
        significant_days['baseline_sentiment'] = significant_days['avg_sentiment'].rolling(
            window=min(7, len(significant_days)), 
            min_periods=1
        ).mean()
        
        # Detect shifts
        for idx, row in significant_days.iterrows():
            if pd.isna(row['baseline_sentiment']):
                continue
                
            sentiment_shift = abs(row['avg_sentiment'] - row['baseline_sentiment'])
            
            if sentiment_shift > self.config.sentiment_shift_threshold:
                anomaly = TemporalAnomaly(
                    anomaly_id=f"sentiment_change_{row['date']}",
                    app_id=row['app_id'],
                    method=TemporalMethod.SENTIMENT_CHANGE,
                    score=sentiment_shift,
                    is_anomaly=True,
                    time_window_start=datetime.combine(row['date'], datetime.min.time()),
                    time_window_end=datetime.combine(row['date'], datetime.max.time()),
                    affected_reviews=row['review_ids'],
                    baseline_value=row['baseline_sentiment'],
                    anomaly_value=row['avg_sentiment'],
                    explanation=f"Sentiment change: {row['avg_sentiment']:.2f} vs baseline {row['baseline_sentiment']:.2f}"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_seasonal_deviations(self, temporal_df: pd.DataFrame) -> List[TemporalAnomaly]:
        """Detect seasonal pattern deviations"""
        anomalies = []
        
        # Group by month and day of month for seasonal patterns
        temporal_df['month_day'] = temporal_df['timestamp'].dt.strftime('%m-%d')
        monthly_patterns = temporal_df.groupby('month_day').agg({
            'rating': 'mean',
            'review_id': 'count'
        }).reset_index()
        
        monthly_patterns.columns = ['month_day', 'avg_rating', 'review_count']
        
        if len(monthly_patterns) < 10:  # Need sufficient seasonal data
            return anomalies
        
        # Calculate seasonal baselines
        seasonal_rating_mean = monthly_patterns['avg_rating'].mean()
        seasonal_rating_std = monthly_patterns['avg_rating'].std()
        
        # Detect deviations
        for _, row in monthly_patterns.iterrows():
            if row['review_count'] < 3:  # Skip days with insufficient data
                continue
                
            deviation = abs(row['avg_rating'] - seasonal_rating_mean) / seasonal_rating_std
            
            if deviation > self.config.seasonal_deviation_threshold:
                # Find the specific date(s) with this pattern
                matching_reviews = temporal_df[
                    temporal_df['month_day'] == row['month_day']
                ]
                
                if len(matching_reviews) > 0:
                    app_id = matching_reviews['app_id'].iloc[0]
                    affected_reviews = matching_reviews['review_id'].tolist()
                    
                    # Use the most recent occurrence
                    latest_date = matching_reviews['timestamp'].max()
                    
                    anomaly = TemporalAnomaly(
                        anomaly_id=f"seasonal_deviation_{row['month_day']}",
                        app_id=app_id,
                        method=TemporalMethod.SEASONAL_DEVIATION,
                        score=deviation,
                        is_anomaly=True,
                        time_window_start=latest_date.replace(hour=0, minute=0, second=0),
                        time_window_end=latest_date.replace(hour=23, minute=59, second=59),
                        affected_reviews=affected_reviews,
                        baseline_value=seasonal_rating_mean,
                        anomaly_value=row['avg_rating'],
                        explanation=f"Seasonal deviation: {deviation:.2f} std devs from seasonal pattern"
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_trend_breaks(self, temporal_df: pd.DataFrame) -> List[TemporalAnomaly]:
        """Detect trend breaks in time series"""
        anomalies = []
        
        # Group by daily windows
        temporal_df['date'] = temporal_df['timestamp'].dt.date
        daily_stats = temporal_df.groupby('date').agg({
            'rating': 'mean',
            'review_id': ['count', lambda x: list(x)],
            'app_id': 'first'
        }).reset_index()
        
        daily_stats.columns = ['date', 'avg_rating', 'review_count', 'review_ids', 'app_id']
        daily_stats = daily_stats.sort_values('date')
        
        if len(daily_stats) < self.config.trend_window_days * 2:
            return anomalies
        
        # Analyze trend breaks using sliding windows
        window_size = self.config.trend_window_days
        
        for i in range(window_size, len(daily_stats) - window_size):
            # Before window
            before_window = daily_stats.iloc[i-window_size:i]
            # After window  
            after_window = daily_stats.iloc[i:i+window_size]
            
            # Calculate trends (simple linear regression slope approximation)
            before_trend = self._calculate_trend(before_window['avg_rating'].values)
            after_trend = self._calculate_trend(after_window['avg_rating'].values)
            
            # Calculate correlation between trends
            if len(before_window) > 1 and len(after_window) > 1:
                try:
                    trend_correlation = np.corrcoef(
                        before_window['avg_rating'].values,
                        after_window['avg_rating'].values
                    )[0, 1]
                    
                    if abs(trend_correlation) < self.config.trend_break_threshold:
                        # Trend break detected
                        break_date = daily_stats.iloc[i]['date']
                        affected_reviews = []
                        
                        # Collect reviews from the break window
                        for _, row in after_window.iterrows():
                            affected_reviews.extend(row['review_ids'])
                        
                        app_id = daily_stats.iloc[i]['app_id']
                        
                        anomaly = TemporalAnomaly(
                            anomaly_id=f"trend_break_{break_date}",
                            app_id=app_id,
                            method=TemporalMethod.TREND_BREAK,
                            score=1.0 - abs(trend_correlation),
                            is_anomaly=True,
                            time_window_start=datetime.combine(after_window.iloc[0]['date'], datetime.min.time()),
                            time_window_end=datetime.combine(after_window.iloc[-1]['date'], datetime.max.time()),
                            affected_reviews=affected_reviews,
                            statistical_significance=abs(trend_correlation),
                            explanation=f"Trend break detected: correlation {trend_correlation:.3f}"
                        )
                        anomalies.append(anomaly)
                
                except Exception as e:
                    logger.debug(f"Trend correlation calculation failed: {e}")
                    continue
        
        return anomalies
    
    def _detect_periodicity_anomalies(self, temporal_df: pd.DataFrame) -> List[TemporalAnomaly]:
        """Detect anomalies in periodic patterns"""
        anomalies = []
        
        # Analyze daily patterns
        temporal_df['hour'] = temporal_df['timestamp'].dt.hour
        hourly_patterns = temporal_df.groupby('hour').size()
        
        # Detect if there are unusual patterns in hourly distribution
        expected_pattern = hourly_patterns.mean()
        pattern_std = hourly_patterns.std()
        
        for hour, count in hourly_patterns.items():
            if abs(count - expected_pattern) > 2 * pattern_std:
                # Find reviews in this unusual hour pattern
                unusual_hour_reviews = temporal_df[
                    temporal_df['hour'] == hour
                ]
                
                if len(unusual_hour_reviews) > 0:
                    app_id = unusual_hour_reviews['app_id'].iloc[0]
                    affected_reviews = unusual_hour_reviews['review_id'].tolist()
                    
                    # Use time window of the first and last occurrence
                    start_time = unusual_hour_reviews['timestamp'].min()
                    end_time = unusual_hour_reviews['timestamp'].max()
                    
                    anomaly = TemporalAnomaly(
                        anomaly_id=f"periodicity_anomaly_hour_{hour}",
                        app_id=app_id,
                        method=TemporalMethod.PERIODICITY_ANOMALY,
                        score=abs(count - expected_pattern) / pattern_std,
                        is_anomaly=True,
                        time_window_start=start_time,
                        time_window_end=end_time,
                        affected_reviews=affected_reviews,
                        baseline_value=expected_pattern,
                        anomaly_value=count,
                        explanation=f"Unusual hourly pattern: {count} reviews at hour {hour} (expected: {expected_pattern:.1f})"
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_time_clustering_anomalies(self, temporal_df: pd.DataFrame) -> List[TemporalAnomaly]:
        """Detect time-based clustering anomalies"""
        anomalies = []
        
        # Convert timestamps to numeric for clustering
        temporal_df['timestamp_numeric'] = temporal_df['timestamp'].astype(np.int64) // 10**9
        
        try:
            from sklearn.cluster import DBSCAN
            
            # Cluster reviews by time
            time_points = temporal_df['timestamp_numeric'].values.reshape(-1, 1)
            
            # Use DBSCAN with time-based eps
            eps_seconds = 3600  # 1 hour
            dbscan = DBSCAN(eps=eps_seconds, min_samples=3)
            clusters = dbscan.fit_predict(time_points)
            
            # Find outlier reviews (cluster = -1)
            outlier_indices = np.where(clusters == -1)[0]
            
            if len(outlier_indices) > 0:
                outlier_reviews = temporal_df.iloc[outlier_indices]
                
                # Group nearby outliers
                outlier_groups = {}
                for idx in outlier_indices:
                    timestamp = temporal_df.iloc[idx]['timestamp']
                    
                    # Find or create group
                    group_key = None
                    for existing_key, existing_timestamp in outlier_groups.items():
                        if abs((timestamp - existing_timestamp).total_seconds()) < 3600:  # 1 hour
                            group_key = existing_key
                            break
                    
                    if group_key is None:
                        group_key = len(outlier_groups)
                        outlier_groups[group_key] = timestamp
                
                # Create anomalies for each group
                for group_id, group_timestamp in outlier_groups.items():
                    group_reviews = outlier_reviews[
                        abs((outlier_reviews['timestamp'] - group_timestamp).dt.total_seconds()) < 3600
                    ]
                    
                    if len(group_reviews) > 0:
                        app_id = group_reviews['app_id'].iloc[0]
                        affected_reviews = group_reviews['review_id'].tolist()
                        
                        start_time = group_reviews['timestamp'].min()
                        end_time = group_reviews['timestamp'].max()
                        
                        anomaly = TemporalAnomaly(
                            anomaly_id=f"time_clustering_anomaly_{group_id}",
                            app_id=app_id,
                            method=TemporalMethod.TIME_CLUSTERING,
                            score=len(group_reviews),
                            is_anomaly=True,
                            time_window_start=start_time,
                            time_window_end=end_time,
                            affected_reviews=affected_reviews,
                            explanation=f"Temporal clustering outlier: {len(group_reviews)} isolated reviews"
                        )
                        anomalies.append(anomaly)
        
        except ImportError:
            logger.warning("sklearn not available for time clustering analysis")
        except Exception as e:
            logger.error(f"Time clustering analysis failed: {e}")
        
        return anomalies
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate simple trend slope"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = values
        
        # Simple linear regression slope
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
        
        return slope
    
    def _deduplicate_anomalies(self, anomalies: List[TemporalAnomaly]) -> List[TemporalAnomaly]:
        """Remove duplicate and overlapping anomalies"""
        if not anomalies:
            return anomalies
        
        # Sort by score (highest first) and time
        anomalies_sorted = sorted(
            anomalies, 
            key=lambda x: (x.score, x.time_window_start), 
            reverse=True
        )
        
        unique_anomalies = []
        
        for anomaly in anomalies_sorted:
            # Check for overlaps with existing anomalies
            is_duplicate = False
            
            for existing in unique_anomalies:
                # Check for time overlap
                if (anomaly.time_window_start <= existing.time_window_end and
                    anomaly.time_window_end >= existing.time_window_start):
                    
                    # Check if same method and app
                    if (anomaly.method == existing.method and 
                        anomaly.app_id == existing.app_id):
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_anomalies.append(anomaly)
        
        return unique_anomalies
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics"""
        return {
            "config": {
                "methods": [m.value for m in self.config.methods],
                "spike_window_hours": self.config.spike_window_hours,
                "volume_spike_multiplier": self.config.volume_spike_multiplier,
                "rating_shift_threshold": self.config.rating_shift_threshold,
                "trend_window_days": self.config.trend_window_days
            }
        }