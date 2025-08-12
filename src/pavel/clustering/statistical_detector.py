"""
Statistical anomaly detection for Google Play reviews.

Implements various statistical methods to detect outliers and anomalies
in review data based on distribution properties, Z-scores, and IQR analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from scipy import stats
from collections import defaultdict
import warnings

from pavel.core.logger import get_logger

logger = get_logger(__name__)

class StatisticalMethod(Enum):
    """Statistical anomaly detection methods"""
    Z_SCORE = "z_score"
    MODIFIED_Z_SCORE = "modified_z_score" 
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    GRUBBS = "grubbs"
    DIXON = "dixon"
    PERCENTILE = "percentile"

@dataclass
class StatisticalConfig:
    """Configuration for statistical anomaly detection"""
    # Z-score thresholds
    z_score_threshold: float = 3.0
    modified_z_score_threshold: float = 3.5
    
    # IQR multiplier
    iqr_multiplier: float = 1.5
    
    # Percentile thresholds
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    
    # Isolation Forest parameters
    isolation_contamination: float = 0.1
    isolation_n_estimators: int = 100
    
    # Grubbs test significance
    grubbs_alpha: float = 0.05
    
    # Minimum sample size for statistical tests
    min_sample_size: int = 10
    
    # Methods to apply
    methods: List[StatisticalMethod] = None
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = [
                StatisticalMethod.Z_SCORE,
                StatisticalMethod.IQR,
                StatisticalMethod.PERCENTILE
            ]

@dataclass
class StatisticalAnomaly:
    """Statistical anomaly detection result"""
    review_id: str
    app_id: str
    method: StatisticalMethod
    score: float
    threshold: float
    is_anomaly: bool
    feature: str
    value: float
    percentile: Optional[float] = None
    explanation: Optional[str] = None

class StatisticalAnomalyDetector:
    """
    Statistical anomaly detector for Google Play reviews.
    
    Detects outliers using various statistical methods:
    - Z-score and Modified Z-score
    - Interquartile Range (IQR)
    - Percentile-based detection
    - Grubbs test for outliers
    - Isolation Forest (if sklearn available)
    
    Features analyzed:
    - Review length
    - Rating distribution
    - Time patterns
    - Language-specific patterns
    """
    
    def __init__(self, config: Optional[StatisticalConfig] = None):
        self.config = config or StatisticalConfig()
        self._sklearn_available = self._check_sklearn()
        
        if StatisticalMethod.ISOLATION_FOREST in self.config.methods and not self._sklearn_available:
            logger.warning("Isolation Forest requires sklearn. Removing from methods.")
            self.config.methods.remove(StatisticalMethod.ISOLATION_FOREST)
    
    def _check_sklearn(self) -> bool:
        """Check if sklearn is available"""
        try:
            import sklearn
            return True
        except ImportError:
            return False
    
    def detect_anomalies(self, reviews: List[Dict[str, Any]]) -> List[StatisticalAnomaly]:
        """
        Detect statistical anomalies in reviews.
        
        Args:
            reviews: List of review dictionaries
            
        Returns:
            List of detected anomalies
        """
        if len(reviews) < self.config.min_sample_size:
            logger.warning(f"Sample size {len(reviews)} below minimum {self.config.min_sample_size}")
            return []
        
        logger.info(f"Analyzing {len(reviews)} reviews for statistical anomalies")
        
        # Convert to DataFrame for easier analysis
        df = self._prepare_dataframe(reviews)
        
        anomalies = []
        
        # Apply each configured method
        for method in self.config.methods:
            method_anomalies = self._apply_method(method, df, reviews)
            anomalies.extend(method_anomalies)
            logger.debug(f"Method {method.value} found {len(method_anomalies)} anomalies")
        
        # Remove duplicates (same review flagged by multiple methods)
        unique_anomalies = self._deduplicate_anomalies(anomalies)
        
        logger.info(f"Found {len(unique_anomalies)} unique statistical anomalies")
        return unique_anomalies
    
    def _prepare_dataframe(self, reviews: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare DataFrame with features for analysis"""
        data = []
        
        for review in reviews:
            row = {
                'review_id': review.get('review_id', ''),
                'app_id': review.get('app_id', ''),
                'content': review.get('content', ''),
                'rating': review.get('rating', 0),
                'locale': review.get('locale', ''),
                'created_at': review.get('created_at'),
                'length': len(review.get('content', '')),
                'word_count': len(review.get('content', '').split()),
                'char_density': self._calculate_char_density(review.get('content', '')),
                'uppercase_ratio': self._calculate_uppercase_ratio(review.get('content', '')),
                'punctuation_ratio': self._calculate_punctuation_ratio(review.get('content', '')),
                'digit_ratio': self._calculate_digit_ratio(review.get('content', ''))
            }
            
            # Add time-based features if timestamp available
            if review.get('created_at'):
                timestamp = self._parse_timestamp(review['created_at'])
                if timestamp:
                    row.update({
                        'hour_of_day': timestamp.hour,
                        'day_of_week': timestamp.weekday(),
                        'is_weekend': timestamp.weekday() >= 5
                    })
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _apply_method(self, method: StatisticalMethod, df: pd.DataFrame, reviews: List[Dict[str, Any]]) -> List[StatisticalAnomaly]:
        """Apply specific statistical method"""
        anomalies = []
        
        if method == StatisticalMethod.Z_SCORE:
            anomalies.extend(self._detect_z_score_anomalies(df))
        elif method == StatisticalMethod.MODIFIED_Z_SCORE:
            anomalies.extend(self._detect_modified_z_score_anomalies(df))
        elif method == StatisticalMethod.IQR:
            anomalies.extend(self._detect_iqr_anomalies(df))
        elif method == StatisticalMethod.PERCENTILE:
            anomalies.extend(self._detect_percentile_anomalies(df))
        elif method == StatisticalMethod.GRUBBS:
            anomalies.extend(self._detect_grubbs_anomalies(df))
        elif method == StatisticalMethod.ISOLATION_FOREST:
            anomalies.extend(self._detect_isolation_forest_anomalies(df))
        
        return anomalies
    
    def _detect_z_score_anomalies(self, df: pd.DataFrame) -> List[StatisticalAnomaly]:
        """Detect anomalies using Z-score method"""
        anomalies = []
        numeric_columns = ['length', 'word_count', 'char_density', 'uppercase_ratio', 
                          'punctuation_ratio', 'digit_ratio', 'rating']
        
        for column in numeric_columns:
            if column not in df.columns or df[column].isna().all():
                continue
                
            values = df[column].dropna()
            if len(values) < self.config.min_sample_size:
                continue
            
            mean_val = values.mean()
            std_val = values.std()
            
            if std_val == 0:  # No variance
                continue
            
            z_scores = np.abs((values - mean_val) / std_val)
            
            for idx, z_score in z_scores.items():
                if z_score > self.config.z_score_threshold:
                    percentile = stats.percentileofscore(values, values.iloc[idx])
                    
                    anomaly = StatisticalAnomaly(
                        review_id=df.iloc[idx]['review_id'],
                        app_id=df.iloc[idx]['app_id'],
                        method=StatisticalMethod.Z_SCORE,
                        score=z_score,
                        threshold=self.config.z_score_threshold,
                        is_anomaly=True,
                        feature=column,
                        value=values.iloc[idx],
                        percentile=percentile,
                        explanation=f"Z-score {z_score:.2f} exceeds threshold {self.config.z_score_threshold}"
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_modified_z_score_anomalies(self, df: pd.DataFrame) -> List[StatisticalAnomaly]:
        """Detect anomalies using Modified Z-score (using median and MAD)"""
        anomalies = []
        numeric_columns = ['length', 'word_count', 'char_density', 'uppercase_ratio', 
                          'punctuation_ratio', 'digit_ratio', 'rating']
        
        for column in numeric_columns:
            if column not in df.columns or df[column].isna().all():
                continue
                
            values = df[column].dropna()
            if len(values) < self.config.min_sample_size:
                continue
            
            median_val = values.median()
            mad = np.median(np.abs(values - median_val))
            
            if mad == 0:  # No variance
                continue
            
            modified_z_scores = 0.6745 * (values - median_val) / mad
            
            for idx, mod_z_score in modified_z_scores.abs().items():
                if mod_z_score > self.config.modified_z_score_threshold:
                    percentile = stats.percentileofscore(values, values.iloc[idx])
                    
                    anomaly = StatisticalAnomaly(
                        review_id=df.iloc[idx]['review_id'],
                        app_id=df.iloc[idx]['app_id'],
                        method=StatisticalMethod.MODIFIED_Z_SCORE,
                        score=mod_z_score,
                        threshold=self.config.modified_z_score_threshold,
                        is_anomaly=True,
                        feature=column,
                        value=values.iloc[idx],
                        percentile=percentile,
                        explanation=f"Modified Z-score {mod_z_score:.2f} exceeds threshold {self.config.modified_z_score_threshold}"
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_iqr_anomalies(self, df: pd.DataFrame) -> List[StatisticalAnomaly]:
        """Detect anomalies using Interquartile Range method"""
        anomalies = []
        numeric_columns = ['length', 'word_count', 'char_density', 'uppercase_ratio', 
                          'punctuation_ratio', 'digit_ratio', 'rating']
        
        for column in numeric_columns:
            if column not in df.columns or df[column].isna().all():
                continue
                
            values = df[column].dropna()
            if len(values) < self.config.min_sample_size:
                continue
            
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:  # No variance
                continue
            
            lower_bound = Q1 - self.config.iqr_multiplier * IQR
            upper_bound = Q3 + self.config.iqr_multiplier * IQR
            
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            
            for idx, value in outliers.items():
                distance_from_bounds = min(abs(value - lower_bound), abs(value - upper_bound))
                score = distance_from_bounds / IQR if IQR > 0 else 0
                percentile = stats.percentileofscore(values, value)
                
                anomaly = StatisticalAnomaly(
                    review_id=df.iloc[idx]['review_id'],
                    app_id=df.iloc[idx]['app_id'],
                    method=StatisticalMethod.IQR,
                    score=score,
                    threshold=self.config.iqr_multiplier,
                    is_anomaly=True,
                    feature=column,
                    value=value,
                    percentile=percentile,
                    explanation=f"Value {value:.2f} outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_percentile_anomalies(self, df: pd.DataFrame) -> List[StatisticalAnomaly]:
        """Detect anomalies using percentile thresholds"""
        anomalies = []
        numeric_columns = ['length', 'word_count', 'char_density', 'uppercase_ratio', 
                          'punctuation_ratio', 'digit_ratio', 'rating']
        
        for column in numeric_columns:
            if column not in df.columns or df[column].isna().all():
                continue
                
            values = df[column].dropna()
            if len(values) < self.config.min_sample_size:
                continue
            
            lower_threshold = values.quantile(self.config.lower_percentile / 100.0)
            upper_threshold = values.quantile(self.config.upper_percentile / 100.0)
            
            outliers = values[(values < lower_threshold) | (values > upper_threshold)]
            
            for idx, value in outliers.items():
                percentile = stats.percentileofscore(values, value)
                score = percentile if percentile > 50 else 100 - percentile
                
                anomaly = StatisticalAnomaly(
                    review_id=df.iloc[idx]['review_id'],
                    app_id=df.iloc[idx]['app_id'],
                    method=StatisticalMethod.PERCENTILE,
                    score=score,
                    threshold=max(self.config.lower_percentile, 100 - self.config.upper_percentile),
                    is_anomaly=True,
                    feature=column,
                    value=value,
                    percentile=percentile,
                    explanation=f"Value at {percentile:.1f} percentile outside [{self.config.lower_percentile}%, {self.config.upper_percentile}%] range"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_grubbs_anomalies(self, df: pd.DataFrame) -> List[StatisticalAnomaly]:
        """Detect anomalies using Grubbs test"""
        anomalies = []
        numeric_columns = ['length', 'word_count', 'char_density', 'rating']
        
        for column in numeric_columns:
            if column not in df.columns or df[column].isna().all():
                continue
                
            values = df[column].dropna()
            if len(values) < self.config.min_sample_size:
                continue
            
            # Grubbs test for outliers
            try:
                mean_val = values.mean()
                std_val = values.std()
                
                if std_val == 0:
                    continue
                
                # Calculate Grubbs statistic for each point
                grubbs_stats = np.abs(values - mean_val) / std_val
                max_grubbs = grubbs_stats.max()
                max_idx = grubbs_stats.idxmax()
                
                # Critical value for Grubbs test
                n = len(values)
                t_critical = stats.t.ppf(1 - self.config.grubbs_alpha / (2 * n), n - 2)
                grubbs_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_critical**2 / (n - 2 + t_critical**2))
                
                if max_grubbs > grubbs_critical:
                    percentile = stats.percentileofscore(values, values.iloc[max_idx])
                    
                    anomaly = StatisticalAnomaly(
                        review_id=df.iloc[max_idx]['review_id'],
                        app_id=df.iloc[max_idx]['app_id'],
                        method=StatisticalMethod.GRUBBS,
                        score=max_grubbs,
                        threshold=grubbs_critical,
                        is_anomaly=True,
                        feature=column,
                        value=values.iloc[max_idx],
                        percentile=percentile,
                        explanation=f"Grubbs statistic {max_grubbs:.3f} exceeds critical value {grubbs_critical:.3f}"
                    )
                    anomalies.append(anomaly)
            
            except Exception as e:
                logger.debug(f"Grubbs test failed for {column}: {e}")
                continue
        
        return anomalies
    
    def _detect_isolation_forest_anomalies(self, df: pd.DataFrame) -> List[StatisticalAnomaly]:
        """Detect anomalies using Isolation Forest"""
        if not self._sklearn_available:
            return []
        
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            # Select numeric features
            numeric_columns = ['length', 'word_count', 'char_density', 'uppercase_ratio', 
                              'punctuation_ratio', 'digit_ratio', 'rating']
            
            available_columns = [col for col in numeric_columns if col in df.columns and not df[col].isna().all()]
            
            if len(available_columns) < 2:
                return []
            
            # Prepare feature matrix
            X = df[available_columns].fillna(df[available_columns].median())
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply Isolation Forest
            iso_forest = IsolationForest(
                contamination=self.config.isolation_contamination,
                n_estimators=self.config.isolation_n_estimators,
                random_state=42
            )
            
            predictions = iso_forest.fit_predict(X_scaled)
            scores = iso_forest.decision_function(X_scaled)
            
            anomalies = []
            for idx, (prediction, score) in enumerate(zip(predictions, scores)):
                if prediction == -1:  # Anomaly detected
                    anomaly = StatisticalAnomaly(
                        review_id=df.iloc[idx]['review_id'],
                        app_id=df.iloc[idx]['app_id'],
                        method=StatisticalMethod.ISOLATION_FOREST,
                        score=abs(score),
                        threshold=0.0,
                        is_anomaly=True,
                        feature="multivariate",
                        value=score,
                        explanation=f"Isolation Forest anomaly score: {score:.3f}"
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Isolation Forest detection failed: {e}")
            return []
    
    def _deduplicate_anomalies(self, anomalies: List[StatisticalAnomaly]) -> List[StatisticalAnomaly]:
        """Remove duplicate anomalies for the same review"""
        seen_reviews = {}
        unique_anomalies = []
        
        # Sort by score (highest first) to keep best anomalies
        anomalies_sorted = sorted(anomalies, key=lambda x: x.score, reverse=True)
        
        for anomaly in anomalies_sorted:
            key = (anomaly.review_id, anomaly.feature)
            if key not in seen_reviews:
                seen_reviews[key] = anomaly
                unique_anomalies.append(anomaly)
        
        return unique_anomalies
    
    def _calculate_char_density(self, text: str) -> float:
        """Calculate character density (chars per word)"""
        if not text:
            return 0.0
        words = text.split()
        if not words:
            return 0.0
        return len(text) / len(words)
    
    def _calculate_uppercase_ratio(self, text: str) -> float:
        """Calculate ratio of uppercase characters"""
        if not text:
            return 0.0
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars == 0:
            return 0.0
        uppercase_chars = sum(1 for c in text if c.isupper())
        return uppercase_chars / alpha_chars
    
    def _calculate_punctuation_ratio(self, text: str) -> float:
        """Calculate ratio of punctuation characters"""
        if not text:
            return 0.0
        punct_chars = sum(1 for c in text if c in '.,!?;:')
        return punct_chars / len(text)
    
    def _calculate_digit_ratio(self, text: str) -> float:
        """Calculate ratio of digit characters"""
        if not text:
            return 0.0
        digit_chars = sum(1 for c in text if c.isdigit())
        return digit_chars / len(text)
    
    def _parse_timestamp(self, timestamp) -> Optional[datetime]:
        """Parse timestamp to datetime object"""
        if isinstance(timestamp, datetime):
            return timestamp
        elif isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                return None
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics"""
        return {
            "config": {
                "z_score_threshold": self.config.z_score_threshold,
                "iqr_multiplier": self.config.iqr_multiplier,
                "methods": [m.value for m in self.config.methods]
            },
            "sklearn_available": self._sklearn_available
        }