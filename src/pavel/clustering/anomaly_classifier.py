"""
Anomaly classification and sensitivity configuration for Google Play reviews.

Classifies detected anomalies by type, severity, and business impact.
Provides configurable sensitivity and threshold settings.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from collections import Counter, defaultdict

from pavel.core.logger import get_logger
from .statistical_detector import StatisticalAnomaly
from .clustering_detector import ClusteringAnomaly
from .semantic_detector import SemanticAnomaly
from .temporal_detector import TemporalAnomaly

logger = get_logger(__name__)

class AnomalyType(Enum):
    """Types of anomalies"""
    # Content anomalies
    SPAM_REVIEW = "spam_review"
    FAKE_REVIEW = "fake_review"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    DUPLICATE_CONTENT = "duplicate_content"
    
    # Rating anomalies
    RATING_MANIPULATION = "rating_manipulation"
    SUDDEN_RATING_DROP = "sudden_rating_drop"
    INCONSISTENT_RATING = "inconsistent_rating"
    
    # Behavioral anomalies
    REVIEW_BOMBING = "review_bombing"
    COORDINATED_ATTACK = "coordinated_attack"
    BOT_ACTIVITY = "bot_activity"
    UNUSUAL_TIMING = "unusual_timing"
    
    # Technical anomalies
    STATISTICAL_OUTLIER = "statistical_outlier"
    SEMANTIC_OUTLIER = "semantic_outlier"
    TEMPORAL_OUTLIER = "temporal_outlier"
    CLUSTERING_OUTLIER = "clustering_outlier"
    
    # Business impact anomalies
    COMPETITOR_MENTION = "competitor_mention"
    FEATURE_REQUEST_SPIKE = "feature_request_spike"
    CRASH_REPORT_SPIKE = "crash_report_spike"
    PAYMENT_ISSUE_SPIKE = "payment_issue_spike"

class AnomalySeverity(Enum):
    """Severity levels for anomalies"""
    CRITICAL = "critical"    # Immediate attention required
    HIGH = "high"           # Requires prompt investigation
    MEDIUM = "medium"       # Should be reviewed soon
    LOW = "low"            # Informational, low priority
    INFO = "info"          # Statistical information

class BusinessImpact(Enum):
    """Business impact levels"""
    HIGH_IMPACT = "high_impact"      # Direct impact on business metrics
    MEDIUM_IMPACT = "medium_impact"  # Potential impact on user experience
    LOW_IMPACT = "low_impact"        # Minimal business impact
    UNKNOWN_IMPACT = "unknown_impact" # Impact not determined

@dataclass
class SensitivityConfig:
    """Configurable sensitivity settings for anomaly detection"""
    # Global sensitivity multiplier (0.5 = less sensitive, 2.0 = more sensitive)
    global_sensitivity: float = 1.0
    
    # Per-type sensitivity overrides
    type_sensitivity: Dict[AnomalyType, float] = None
    
    # Statistical thresholds
    statistical_z_score_threshold: float = 3.0
    statistical_iqr_multiplier: float = 1.5
    
    # Clustering thresholds  
    clustering_contamination: float = 0.1
    clustering_eps: float = 0.5
    
    # Semantic thresholds
    semantic_cosine_threshold: float = 0.7
    semantic_density_threshold: float = 0.5
    
    # Temporal thresholds
    temporal_volume_spike_multiplier: float = 3.0
    temporal_rating_shift_threshold: float = 1.0
    
    # Business impact thresholds
    review_count_threshold: int = 10
    time_window_threshold_hours: int = 24
    
    # Severity scoring weights
    severity_score_thresholds: Dict[AnomalySeverity, float] = None
    
    def __post_init__(self):
        if self.type_sensitivity is None:
            self.type_sensitivity = {
                AnomalyType.REVIEW_BOMBING: 1.5,
                AnomalyType.COORDINATED_ATTACK: 2.0,
                AnomalyType.RATING_MANIPULATION: 1.8,
                AnomalyType.BOT_ACTIVITY: 1.7,
                AnomalyType.SPAM_REVIEW: 1.3,
                AnomalyType.CRASH_REPORT_SPIKE: 1.4,
                AnomalyType.PAYMENT_ISSUE_SPIKE: 1.6
            }
        
        if self.severity_score_thresholds is None:
            self.severity_score_thresholds = {
                AnomalySeverity.CRITICAL: 8.0,
                AnomalySeverity.HIGH: 6.0,
                AnomalySeverity.MEDIUM: 4.0,
                AnomalySeverity.LOW: 2.0,
                AnomalySeverity.INFO: 0.0
            }

@dataclass
class ClassifiedAnomaly:
    """Classified anomaly with type, severity and business context"""
    # Original anomaly data
    original_anomaly: Union[StatisticalAnomaly, ClusteringAnomaly, SemanticAnomaly, TemporalAnomaly]
    
    # Classification results
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    business_impact: BusinessImpact
    
    # Scoring
    confidence_score: float  # 0.0 - 1.0
    severity_score: float    # Numeric severity score
    
    # Context
    affected_reviews_count: int
    time_window_hours: Optional[float] = None
    related_anomalies: List[str] = None
    
    # Explanations
    classification_reason: str = ""
    recommended_action: str = ""
    
    # Metadata
    classified_at: datetime = None
    
    def __post_init__(self):
        if self.classified_at is None:
            self.classified_at = datetime.utcnow()
        
        if self.related_anomalies is None:
            self.related_anomalies = []

class AnomalyClassifier:
    """
    Classifier for detected anomalies with configurable sensitivity.
    
    Analyzes raw anomaly detections and classifies them by:
    - Type (spam, fake reviews, rating manipulation, etc.)
    - Severity (critical, high, medium, low, info)
    - Business impact (high, medium, low, unknown)
    - Confidence and scoring
    
    Provides configurable sensitivity settings for different types
    of anomalies and business contexts.
    """
    
    def __init__(self, config: Optional[SensitivityConfig] = None):
        self.config = config or SensitivityConfig()
        
        # Classification rules and patterns
        self._init_classification_rules()
    
    def _init_classification_rules(self):
        """Initialize classification rules and patterns"""
        # Content patterns for different anomaly types
        self.spam_patterns = [
            'click here', 'visit our', 'download now', 'free money',
            'guaranteed', 'limited time', 'act now', 'no risk'
        ]
        
        self.fake_review_indicators = [
            'amazing app', 'best app ever', 'five stars', '5 stars',
            'highly recommend', 'must download', 'perfect app'
        ]
        
        self.inappropriate_patterns = [
            'hate speech', 'offensive', 'harassment', 'discrimination',
            'violence', 'illegal', 'explicit', 'inappropriate'
        ]
        
        self.competitor_patterns = [
            'better than', 'compared to', 'alternative to', 'instead use',
            'try this instead', 'switch to', 'much better'
        ]
        
        self.crash_patterns = [
            'crash', 'crashes', 'freeze', 'freezes', 'stuck', 'hang',
            'not responding', 'force close', 'error', 'bug'
        ]
        
        self.payment_patterns = [
            'payment', 'charge', 'money', 'refund', 'billing', 'subscription',
            'cancel', 'fraud', 'unauthorized', 'stolen'
        ]
    
    def classify_anomalies(self, 
                          anomalies: List[Union[StatisticalAnomaly, ClusteringAnomaly, 
                                              SemanticAnomaly, TemporalAnomaly]]) -> List[ClassifiedAnomaly]:
        """
        Classify a list of detected anomalies.
        
        Args:
            anomalies: List of detected anomalies from various detectors
            
        Returns:
            List of classified anomalies with type, severity, and context
        """
        logger.info(f"Classifying {len(anomalies)} detected anomalies")
        
        classified_anomalies = []
        
        for anomaly in anomalies:
            try:
                classified = self._classify_single_anomaly(anomaly)
                if classified:
                    classified_anomalies.append(classified)
            except Exception as e:
                logger.error(f"Failed to classify anomaly: {e}")
                continue
        
        # Post-process: find related anomalies and adjust severities
        classified_anomalies = self._post_process_classifications(classified_anomalies)
        
        logger.info(f"Successfully classified {len(classified_anomalies)} anomalies")
        return classified_anomalies
    
    def _classify_single_anomaly(self, 
                                anomaly: Union[StatisticalAnomaly, ClusteringAnomaly, 
                                             SemanticAnomaly, TemporalAnomaly]) -> Optional[ClassifiedAnomaly]:
        """Classify a single anomaly"""
        
        # Determine base type from detector type
        base_type = self._get_base_type_from_detector(anomaly)
        
        # Analyze content for more specific classification
        content_type = self._analyze_content_for_type(anomaly)
        
        # Choose final type (content-based overrides detector-based)
        final_type = content_type if content_type != AnomalyType.STATISTICAL_OUTLIER else base_type
        
        # Calculate severity
        severity, severity_score = self._calculate_severity(anomaly, final_type)
        
        # Determine business impact
        business_impact = self._determine_business_impact(anomaly, final_type)
        
        # Calculate confidence
        confidence_score = self._calculate_confidence(anomaly, final_type)
        
        # Get affected reviews count
        affected_count = self._get_affected_reviews_count(anomaly)
        
        # Calculate time window
        time_window = self._calculate_time_window(anomaly)
        
        # Generate explanations
        reason, action = self._generate_explanations(anomaly, final_type, severity)
        
        classified = ClassifiedAnomaly(
            original_anomaly=anomaly,
            anomaly_type=final_type,
            severity=severity,
            business_impact=business_impact,
            confidence_score=confidence_score,
            severity_score=severity_score,
            affected_reviews_count=affected_count,
            time_window_hours=time_window,
            classification_reason=reason,
            recommended_action=action
        )
        
        return classified
    
    def _get_base_type_from_detector(self, anomaly) -> AnomalyType:
        """Get base anomaly type from detector type"""
        if isinstance(anomaly, StatisticalAnomaly):
            return AnomalyType.STATISTICAL_OUTLIER
        elif isinstance(anomaly, ClusteringAnomaly):
            return AnomalyType.CLUSTERING_OUTLIER
        elif isinstance(anomaly, SemanticAnomaly):
            return AnomalyType.SEMANTIC_OUTLIER
        elif isinstance(anomaly, TemporalAnomaly):
            # More specific classification for temporal anomalies
            if hasattr(anomaly, 'method'):
                method_str = str(anomaly.method.value) if hasattr(anomaly.method, 'value') else str(anomaly.method)
                if 'volume_spike' in method_str:
                    return AnomalyType.REVIEW_BOMBING
                elif 'rating_shift' in method_str:
                    return AnomalyType.SUDDEN_RATING_DROP
                elif 'time_clustering' in method_str:
                    return AnomalyType.UNUSUAL_TIMING
            return AnomalyType.TEMPORAL_OUTLIER
        else:
            return AnomalyType.STATISTICAL_OUTLIER
    
    def _analyze_content_for_type(self, anomaly) -> AnomalyType:
        """Analyze review content to determine more specific anomaly type"""
        # Get review content
        content = ""
        if hasattr(anomaly, 'text'):
            content = anomaly.text.lower()
        elif hasattr(anomaly, 'original_anomaly') and hasattr(anomaly.original_anomaly, 'text'):
            content = anomaly.original_anomaly.text.lower()
        
        # If no content available, return generic type
        if not content:
            return AnomalyType.STATISTICAL_OUTLIER
        
        # Check for spam patterns
        if any(pattern in content for pattern in self.spam_patterns):
            return AnomalyType.SPAM_REVIEW
        
        # Check for fake review indicators
        fake_indicators = sum(1 for pattern in self.fake_review_indicators if pattern in content)
        if fake_indicators >= 2:  # Multiple indicators suggest fake review
            return AnomalyType.FAKE_REVIEW
        
        # Check for inappropriate content
        if any(pattern in content for pattern in self.inappropriate_patterns):
            return AnomalyType.INAPPROPRIATE_CONTENT
        
        # Check for competitor mentions
        if any(pattern in content for pattern in self.competitor_patterns):
            return AnomalyType.COMPETITOR_MENTION
        
        # Check for crash reports
        crash_mentions = sum(1 for pattern in self.crash_patterns if pattern in content)
        if crash_mentions >= 2:
            return AnomalyType.CRASH_REPORT_SPIKE
        
        # Check for payment issues
        payment_mentions = sum(1 for pattern in self.payment_patterns if pattern in content)
        if payment_mentions >= 2:
            return AnomalyType.PAYMENT_ISSUE_SPIKE
        
        # Default to statistical outlier if no specific patterns found
        return AnomalyType.STATISTICAL_OUTLIER
    
    def _calculate_severity(self, anomaly, anomaly_type: AnomalyType) -> Tuple[AnomalySeverity, float]:
        """Calculate severity and numeric score for anomaly"""
        
        # Base score from anomaly strength
        base_score = float(getattr(anomaly, 'score', 1.0))
        
        # Apply global sensitivity
        base_score *= self.config.global_sensitivity
        
        # Apply type-specific sensitivity
        if anomaly_type in self.config.type_sensitivity:
            base_score *= self.config.type_sensitivity[anomaly_type]
        
        # Bonus for high-impact types
        if anomaly_type in [
            AnomalyType.REVIEW_BOMBING, 
            AnomalyType.COORDINATED_ATTACK,
            AnomalyType.RATING_MANIPULATION
        ]:
            base_score *= 1.5
        
        # Bonus for multiple affected reviews
        affected_count = self._get_affected_reviews_count(anomaly)
        if affected_count > 10:
            base_score *= 1.2
        elif affected_count > 50:
            base_score *= 1.5
        
        # Determine severity level
        if base_score >= self.config.severity_score_thresholds[AnomalySeverity.CRITICAL]:
            return AnomalySeverity.CRITICAL, base_score
        elif base_score >= self.config.severity_score_thresholds[AnomalySeverity.HIGH]:
            return AnomalySeverity.HIGH, base_score
        elif base_score >= self.config.severity_score_thresholds[AnomalySeverity.MEDIUM]:
            return AnomalySeverity.MEDIUM, base_score
        elif base_score >= self.config.severity_score_thresholds[AnomalySeverity.LOW]:
            return AnomalySeverity.LOW, base_score
        else:
            return AnomalySeverity.INFO, base_score
    
    def _determine_business_impact(self, anomaly, anomaly_type: AnomalyType) -> BusinessImpact:
        """Determine business impact level"""
        
        # High impact types
        if anomaly_type in [
            AnomalyType.RATING_MANIPULATION,
            AnomalyType.REVIEW_BOMBING,
            AnomalyType.COORDINATED_ATTACK,
            AnomalyType.SUDDEN_RATING_DROP
        ]:
            return BusinessImpact.HIGH_IMPACT
        
        # Medium impact types
        if anomaly_type in [
            AnomalyType.SPAM_REVIEW,
            AnomalyType.FAKE_REVIEW,
            AnomalyType.CRASH_REPORT_SPIKE,
            AnomalyType.PAYMENT_ISSUE_SPIKE,
            AnomalyType.BOT_ACTIVITY
        ]:
            return BusinessImpact.MEDIUM_IMPACT
        
        # Low impact types
        if anomaly_type in [
            AnomalyType.INAPPROPRIATE_CONTENT,
            AnomalyType.COMPETITOR_MENTION,
            AnomalyType.FEATURE_REQUEST_SPIKE,
            AnomalyType.UNUSUAL_TIMING
        ]:
            return BusinessImpact.LOW_IMPACT
        
        # Unknown impact for technical outliers
        return BusinessImpact.UNKNOWN_IMPACT
    
    def _calculate_confidence(self, anomaly, anomaly_type: AnomalyType) -> float:
        """Calculate confidence score for classification"""
        
        # Base confidence from anomaly score
        base_confidence = min(1.0, float(getattr(anomaly, 'score', 0.5)) / 5.0)
        
        # Higher confidence for content-based classifications
        if anomaly_type not in [
            AnomalyType.STATISTICAL_OUTLIER,
            AnomalyType.CLUSTERING_OUTLIER,
            AnomalyType.SEMANTIC_OUTLIER,
            AnomalyType.TEMPORAL_OUTLIER
        ]:
            base_confidence *= 1.2
        
        # Lower confidence for statistical outliers
        if anomaly_type == AnomalyType.STATISTICAL_OUTLIER:
            base_confidence *= 0.7
        
        # Adjust based on sample size
        affected_count = self._get_affected_reviews_count(anomaly)
        if affected_count > 5:
            base_confidence *= 1.1
        elif affected_count < 2:
            base_confidence *= 0.8
        
        return min(1.0, base_confidence)
    
    def _get_affected_reviews_count(self, anomaly) -> int:
        """Get number of affected reviews"""
        if hasattr(anomaly, 'affected_reviews') and anomaly.affected_reviews:
            return len(anomaly.affected_reviews)
        elif hasattr(anomaly, 'review_id'):
            return 1
        else:
            return 1
    
    def _calculate_time_window(self, anomaly) -> Optional[float]:
        """Calculate time window in hours for temporal anomalies"""
        if hasattr(anomaly, 'time_window_start') and hasattr(anomaly, 'time_window_end'):
            delta = anomaly.time_window_end - anomaly.time_window_start
            return delta.total_seconds() / 3600.0
        return None
    
    def _generate_explanations(self, anomaly, anomaly_type: AnomalyType, 
                              severity: AnomalySeverity) -> Tuple[str, str]:
        """Generate classification reason and recommended action"""
        
        # Classification reason
        reason_parts = []
        
        # Add detector information
        detector_type = type(anomaly).__name__
        reason_parts.append(f"Detected by {detector_type}")
        
        # Add anomaly-specific details
        if hasattr(anomaly, 'explanation') and anomaly.explanation:
            reason_parts.append(anomaly.explanation)
        
        # Add type-specific reasoning
        if anomaly_type == AnomalyType.SPAM_REVIEW:
            reason_parts.append("Contains spam-like promotional content")
        elif anomaly_type == AnomalyType.FAKE_REVIEW:
            reason_parts.append("Shows characteristics of fake reviews")
        elif anomaly_type == AnomalyType.REVIEW_BOMBING:
            reason_parts.append("Part of unusual volume spike pattern")
        elif anomaly_type == AnomalyType.RATING_MANIPULATION:
            reason_parts.append("Inconsistent with typical rating patterns")
        
        reason = "; ".join(reason_parts)
        
        # Recommended action based on severity and type
        if severity == AnomalySeverity.CRITICAL:
            if anomaly_type in [AnomalyType.REVIEW_BOMBING, AnomalyType.COORDINATED_ATTACK]:
                action = "Immediately investigate for coordinated attack; consider temporary review restrictions"
            else:
                action = "Requires immediate investigation and potential intervention"
        elif severity == AnomalySeverity.HIGH:
            if anomaly_type == AnomalyType.RATING_MANIPULATION:
                action = "Review for rating manipulation; consider rating adjustments"
            elif anomaly_type in [AnomalyType.SPAM_REVIEW, AnomalyType.FAKE_REVIEW]:
                action = "Flag for manual review; consider content moderation action"
            else:
                action = "Investigate promptly; monitor for escalation"
        elif severity == AnomalySeverity.MEDIUM:
            action = "Schedule for review; monitor trend development"
        elif severity == AnomalySeverity.LOW:
            action = "Add to monitoring dashboard; review during regular cycles"
        else:
            action = "Log for statistical analysis; no immediate action required"
        
        return reason, action
    
    def _post_process_classifications(self, classified_anomalies: List[ClassifiedAnomaly]) -> List[ClassifiedAnomaly]:
        """Post-process classifications to find relationships and adjust severities"""
        
        # Group by app and time window for relationship analysis
        app_time_groups = defaultdict(list)
        
        for anomaly in classified_anomalies:
            app_id = getattr(anomaly.original_anomaly, 'app_id', 'unknown')
            
            # Create time bucket (hour-level)
            if hasattr(anomaly.original_anomaly, 'time_window_start'):
                time_bucket = anomaly.original_anomaly.time_window_start.strftime('%Y-%m-%d-%H')
            else:
                time_bucket = 'no_time'
            
            group_key = f"{app_id}_{time_bucket}"
            app_time_groups[group_key].append(anomaly)
        
        # Find coordinated attacks (multiple anomaly types in same time window)
        for group_key, group_anomalies in app_time_groups.items():
            if len(group_anomalies) >= 3:  # Multiple anomalies in same window
                
                # Check for coordinated attack pattern
                anomaly_types = [a.anomaly_type for a in group_anomalies]
                type_counts = Counter(anomaly_types)
                
                # If we have volume spike + rating manipulation + semantic outliers
                has_volume_spike = AnomalyType.REVIEW_BOMBING in anomaly_types
                has_rating_issues = any(t in anomaly_types for t in [
                    AnomalyType.SUDDEN_RATING_DROP, 
                    AnomalyType.RATING_MANIPULATION,
                    AnomalyType.INCONSISTENT_RATING
                ])
                has_content_issues = any(t in anomaly_types for t in [
                    AnomalyType.SPAM_REVIEW,
                    AnomalyType.FAKE_REVIEW,
                    AnomalyType.SEMANTIC_OUTLIER
                ])
                
                # Mark as coordinated attack if multiple attack vectors
                if sum([has_volume_spike, has_rating_issues, has_content_issues]) >= 2:
                    for anomaly in group_anomalies:
                        anomaly.anomaly_type = AnomalyType.COORDINATED_ATTACK
                        anomaly.severity = AnomalySeverity.CRITICAL
                        anomaly.severity_score *= 2.0
                        anomaly.business_impact = BusinessImpact.HIGH_IMPACT
                        anomaly.classification_reason += "; Part of coordinated attack pattern"
                        anomaly.recommended_action = "URGENT: Coordinated attack detected - immediate response required"
                        
                        # Add related anomalies
                        related_ids = [
                            getattr(other.original_anomaly, 'review_id', 'unknown') 
                            for other in group_anomalies if other != anomaly
                        ]
                        anomaly.related_anomalies = related_ids
        
        return classified_anomalies
    
    def configure_sensitivity(self, 
                            global_sensitivity: Optional[float] = None,
                            type_sensitivities: Optional[Dict[AnomalyType, float]] = None,
                            severity_thresholds: Optional[Dict[AnomalySeverity, float]] = None):
        """
        Update sensitivity configuration.
        
        Args:
            global_sensitivity: Global sensitivity multiplier
            type_sensitivities: Per-type sensitivity overrides
            severity_thresholds: Severity score thresholds
        """
        if global_sensitivity is not None:
            self.config.global_sensitivity = global_sensitivity
            logger.info(f"Updated global sensitivity to {global_sensitivity}")
        
        if type_sensitivities:
            self.config.type_sensitivity.update(type_sensitivities)
            logger.info(f"Updated type sensitivities for {len(type_sensitivities)} types")
        
        if severity_thresholds:
            self.config.severity_score_thresholds.update(severity_thresholds)
            logger.info(f"Updated severity thresholds for {len(severity_thresholds)} levels")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get classifier statistics"""
        return {
            "config": {
                "global_sensitivity": self.config.global_sensitivity,
                "type_sensitivity_count": len(self.config.type_sensitivity),
                "severity_thresholds": {
                    k.value: v for k, v in self.config.severity_score_thresholds.items()
                }
            },
            "classification_rules": {
                "spam_patterns": len(self.spam_patterns),
                "fake_review_indicators": len(self.fake_review_indicators),
                "crash_patterns": len(self.crash_patterns),
                "payment_patterns": len(self.payment_patterns)
            }
        }