"""
Complete anomaly detection pipeline for Google Play reviews.

Integrates all detection methods and provides end-to-end anomaly detection
with classification, severity assessment, and business impact analysis.
"""

import asyncio
import time
import numpy as np
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict, Counter

from pavel.core.logger import get_logger
from pavel.core.config import get_config

# Import Stage 3-4 components
from ..embeddings.embedding_pipeline import EmbeddingPipeline
from ..embeddings.semantic_search import SemanticSearchEngine

# Import anomaly detectors
from .statistical_detector import StatisticalAnomalyDetector, StatisticalConfig
from .clustering_detector import ClusteringAnomalyDetector, ClusteringConfig
from .semantic_detector import SemanticAnomalyDetector, SemanticConfig
from .temporal_detector import TemporalAnomalyDetector, TemporalConfig
from .anomaly_classifier import AnomalyClassifier, SensitivityConfig, ClassifiedAnomaly

logger = get_logger(__name__)

@dataclass
class DetectionConfig:
    """Configuration for anomaly detection pipeline"""
    # Enable/disable detection methods
    enable_statistical: bool = True
    enable_clustering: bool = True
    enable_semantic: bool = True
    enable_temporal: bool = True
    
    # Method-specific configurations
    statistical_config: Optional[StatisticalConfig] = None
    clustering_config: Optional[ClusteringConfig] = None
    semantic_config: Optional[SemanticConfig] = None
    temporal_config: Optional[TemporalConfig] = None
    sensitivity_config: Optional[SensitivityConfig] = None
    
    # Pipeline settings
    min_reviews_for_detection: int = 10
    max_concurrent_detectors: int = 4
    
    # Output filtering
    min_severity_level: str = "low"  # info, low, medium, high, critical
    max_anomalies_per_app: int = 100
    
    # Integration with embedding pipeline
    use_existing_embeddings: bool = True
    reprocess_embeddings: bool = False

@dataclass
class AnomalyResult:
    """Complete anomaly detection result"""
    app_id: str
    total_reviews_analyzed: int
    processing_time: float
    
    # Detection results
    statistical_anomalies: int
    clustering_anomalies: int
    semantic_anomalies: int
    temporal_anomalies: int
    
    # Classification results
    classified_anomalies: List[ClassifiedAnomaly]
    
    # Summary statistics
    severity_distribution: Dict[str, int]
    type_distribution: Dict[str, int]
    business_impact_distribution: Dict[str, int]
    
    # Recommendations
    critical_issues: List[ClassifiedAnomaly]
    recommended_actions: List[str]
    
    # Processing metadata
    detection_methods_used: List[str]
    embedding_model_used: Optional[str] = None
    processed_at: datetime = None
    
    def __post_init__(self):
        if self.processed_at is None:
            self.processed_at = datetime.utcnow()

class AnomalyDetectionPipeline:
    """
    Complete anomaly detection pipeline for Google Play reviews.
    
    Integrates:
    - Stage 3: Text preprocessing
    - Stage 4: Vector embeddings and semantic search
    - Stage 5: Multi-method anomaly detection
    - Classification and business impact analysis
    
    Provides end-to-end anomaly detection from raw reviews to
    actionable insights with configurable sensitivity and methods.
    """
    
    def __init__(self, 
                 config: Optional[DetectionConfig] = None,
                 embedding_pipeline: Optional[EmbeddingPipeline] = None):
        self.config = config or DetectionConfig()
        
        # Initialize embedding pipeline (Stage 3-4 integration)
        self.embedding_pipeline = embedding_pipeline or EmbeddingPipeline()
        
        # Initialize anomaly detectors
        self._init_detectors()
        
        # Initialize classifier
        self.classifier = AnomalyClassifier(self.config.sensitivity_config)
        
        # Statistics tracking
        self.processing_stats = {
            'total_reviews_processed': 0,
            'total_anomalies_detected': 0,
            'detection_runs': 0,
            'average_processing_time': 0.0
        }
    
    def _init_detectors(self):
        """Initialize all anomaly detectors"""
        self.detectors = {}
        
        if self.config.enable_statistical:
            self.detectors['statistical'] = StatisticalAnomalyDetector(
                self.config.statistical_config or StatisticalConfig()
            )
        
        if self.config.enable_clustering:
            self.detectors['clustering'] = ClusteringAnomalyDetector(
                self.config.clustering_config or ClusteringConfig()
            )
        
        if self.config.enable_semantic:
            semantic_detector = SemanticAnomalyDetector(
                self.config.semantic_config or SemanticConfig()
            )
            # Integrate with embedding pipeline
            semantic_detector.set_embedding_components(
                self.embedding_pipeline.embedding_generator,
                self.embedding_pipeline.vector_store
            )
            self.detectors['semantic'] = semantic_detector
        
        if self.config.enable_temporal:
            self.detectors['temporal'] = TemporalAnomalyDetector(
                self.config.temporal_config or TemporalConfig()
            )
        
        logger.info(f"Initialized {len(self.detectors)} anomaly detectors")
    
    async def detect_anomalies(self, 
                              app_id: str,
                              reviews: List[Dict[str, Any]],
                              embeddings: Optional[List[np.ndarray]] = None) -> AnomalyResult:
        """
        Detect anomalies in app reviews using all configured methods.
        
        Args:
            app_id: Application identifier
            reviews: List of review dictionaries
            embeddings: Optional precomputed embeddings
            
        Returns:
            Complete anomaly detection result
        """
        if len(reviews) < self.config.min_reviews_for_detection:
            logger.warning(f"Insufficient reviews ({len(reviews)}) for anomaly detection")
            return self._create_empty_result(app_id, len(reviews))
        
        logger.info(f"Starting anomaly detection for {app_id}: {len(reviews)} reviews")
        start_time = time.time()
        
        # Step 1: Get or generate embeddings if needed
        if not embeddings and (self.config.enable_semantic or self.config.use_existing_embeddings):
            embeddings = await self._get_embeddings(reviews)
        
        # Step 2: Run all detection methods concurrently
        detection_results = await self._run_detections(reviews, embeddings)
        
        # Step 3: Classify all detected anomalies
        all_anomalies = []
        for method, anomalies in detection_results.items():
            all_anomalies.extend(anomalies)
        
        classified_anomalies = self.classifier.classify_anomalies(all_anomalies)
        
        # Step 4: Filter by severity and limits
        filtered_anomalies = self._filter_anomalies(classified_anomalies)
        
        # Step 5: Generate summary and recommendations
        processing_time = time.time() - start_time
        result = self._create_result(
            app_id, reviews, detection_results, filtered_anomalies, processing_time
        )
        
        # Update statistics
        self._update_statistics(result)
        
        logger.info(f"Anomaly detection complete for {app_id}: "
                   f"{len(filtered_anomalies)} classified anomalies in {processing_time:.2f}s")
        
        return result
    
    async def _get_embeddings(self, reviews: List[Dict[str, Any]]) -> Optional[List[np.ndarray]]:
        """Get embeddings from existing pipeline or generate new ones"""
        try:
            if self.config.use_existing_embeddings and not self.config.reprocess_embeddings:
                # Try to get existing embeddings from vector store
                existing_embeddings = []
                
                for review in reviews:
                    review_id = review.get('review_id', '')
                    app_id = review.get('app_id', '')
                    
                    # Try to find existing embedding
                    vector_id = f"{app_id}_{review_id}"
                    existing = self.embedding_pipeline.vector_store.get_embedding(vector_id)
                    
                    if existing and 'embedding' in existing:
                        existing_embeddings.append(np.array(existing['embedding']))
                    else:
                        # Generate new embedding for missing reviews
                        text = review.get('content', '')
                        if text:
                            result = self.embedding_pipeline.embedding_generator.generate_single(text)
                            existing_embeddings.append(result.embedding)
                        else:
                            # Zero embedding for empty content
                            zero_emb = np.zeros(384)  # Default dimension
                            existing_embeddings.append(zero_emb)
                
                return existing_embeddings
            
            else:
                # Generate new embeddings
                logger.info("Generating new embeddings for anomaly detection")
                texts = [review.get('content', '') for review in reviews]
                
                embedding_results = await self.embedding_pipeline.embedding_generator.generate_batch_async(texts)
                return [result.embedding for result in embedding_results]
        
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            return None
    
    async def _run_detections(self, 
                             reviews: List[Dict[str, Any]], 
                             embeddings: Optional[List[np.ndarray]]) -> Dict[str, List[Any]]:
        """Run all detection methods concurrently"""
        detection_tasks = {}
        
        # Create detection tasks
        if 'statistical' in self.detectors:
            detection_tasks['statistical'] = asyncio.create_task(
                self._run_statistical_detection(reviews)
            )
        
        if 'clustering' in self.detectors:
            detection_tasks['clustering'] = asyncio.create_task(
                self._run_clustering_detection(reviews)
            )
        
        if 'semantic' in self.detectors and embeddings:
            detection_tasks['semantic'] = asyncio.create_task(
                self._run_semantic_detection(reviews, embeddings)
            )
        
        if 'temporal' in self.detectors:
            detection_tasks['temporal'] = asyncio.create_task(
                self._run_temporal_detection(reviews)
            )
        
        # Wait for all detections to complete
        results = {}
        for method, task in detection_tasks.items():
            try:
                results[method] = await task
                logger.debug(f"{method} detection found {len(results[method])} anomalies")
            except Exception as e:
                logger.error(f"{method} detection failed: {e}")
                results[method] = []
        
        return results
    
    async def _run_statistical_detection(self, reviews: List[Dict[str, Any]]) -> List[Any]:
        """Run statistical anomaly detection"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.detectors['statistical'].detect_anomalies, 
            reviews
        )
    
    async def _run_clustering_detection(self, reviews: List[Dict[str, Any]]) -> List[Any]:
        """Run clustering anomaly detection"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.detectors['clustering'].detect_anomalies, 
            reviews
        )
    
    async def _run_semantic_detection(self, 
                                     reviews: List[Dict[str, Any]], 
                                     embeddings: List[np.ndarray]) -> List[Any]:
        """Run semantic anomaly detection"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.detectors['semantic'].detect_anomalies, 
            reviews, 
            embeddings
        )
    
    async def _run_temporal_detection(self, reviews: List[Dict[str, Any]]) -> List[Any]:
        """Run temporal anomaly detection"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.detectors['temporal'].detect_anomalies, 
            reviews
        )
    
    def _filter_anomalies(self, classified_anomalies: List[ClassifiedAnomaly]) -> List[ClassifiedAnomaly]:
        """Filter anomalies by severity and other criteria"""
        severity_order = {
            'info': 0, 'low': 1, 'medium': 2, 'high': 3, 'critical': 4
        }
        
        min_severity_score = severity_order.get(self.config.min_severity_level.lower(), 0)
        
        filtered = []
        for anomaly in classified_anomalies:
            # Filter by severity
            anomaly_severity_score = severity_order.get(anomaly.severity.value, 0)
            if anomaly_severity_score >= min_severity_score:
                filtered.append(anomaly)
        
        # Sort by severity score (highest first) and limit
        filtered.sort(key=lambda x: x.severity_score, reverse=True)
        
        if len(filtered) > self.config.max_anomalies_per_app:
            filtered = filtered[:self.config.max_anomalies_per_app]
            logger.info(f"Limited anomalies to {self.config.max_anomalies_per_app} per app")
        
        return filtered
    
    def _create_result(self, 
                      app_id: str,
                      reviews: List[Dict[str, Any]],
                      detection_results: Dict[str, List[Any]],
                      classified_anomalies: List[ClassifiedAnomaly],
                      processing_time: float) -> AnomalyResult:
        """Create comprehensive anomaly detection result"""
        
        # Count anomalies by detection method
        statistical_count = len(detection_results.get('statistical', []))
        clustering_count = len(detection_results.get('clustering', []))
        semantic_count = len(detection_results.get('semantic', []))
        temporal_count = len(detection_results.get('temporal', []))
        
        # Analyze classifications
        severity_dist = Counter(a.severity.value for a in classified_anomalies)
        type_dist = Counter(a.anomaly_type.value for a in classified_anomalies)
        impact_dist = Counter(a.business_impact.value for a in classified_anomalies)
        
        # Find critical issues
        critical_issues = [a for a in classified_anomalies 
                          if a.severity.value in ['critical', 'high']]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(critical_issues, classified_anomalies)
        
        # Get embedding model info
        embedding_model = None
        if hasattr(self.embedding_pipeline.embedding_generator, 'model_name'):
            embedding_model = self.embedding_pipeline.embedding_generator.model_name
        
        return AnomalyResult(
            app_id=app_id,
            total_reviews_analyzed=len(reviews),
            processing_time=processing_time,
            statistical_anomalies=statistical_count,
            clustering_anomalies=clustering_count,
            semantic_anomalies=semantic_count,
            temporal_anomalies=temporal_count,
            classified_anomalies=classified_anomalies,
            severity_distribution=dict(severity_dist),
            type_distribution=dict(type_dist),
            business_impact_distribution=dict(impact_dist),
            critical_issues=critical_issues,
            recommended_actions=recommendations,
            detection_methods_used=list(detection_results.keys()),
            embedding_model_used=embedding_model
        )
    
    def _create_empty_result(self, app_id: str, review_count: int) -> AnomalyResult:
        """Create empty result for insufficient data"""
        return AnomalyResult(
            app_id=app_id,
            total_reviews_analyzed=review_count,
            processing_time=0.0,
            statistical_anomalies=0,
            clustering_anomalies=0,
            semantic_anomalies=0,
            temporal_anomalies=0,
            classified_anomalies=[],
            severity_distribution={},
            type_distribution={},
            business_impact_distribution={},
            critical_issues=[],
            recommended_actions=["Insufficient data for anomaly detection"],
            detection_methods_used=[]
        )
    
    def _generate_recommendations(self, 
                                 critical_issues: List[ClassifiedAnomaly],
                                 all_anomalies: List[ClassifiedAnomaly]) -> List[str]:
        """Generate actionable recommendations based on detected anomalies"""
        recommendations = []
        
        # Critical issue recommendations
        if len(critical_issues) > 0:
            coordinated_attacks = [a for a in critical_issues 
                                 if a.anomaly_type.value == 'coordinated_attack']
            if coordinated_attacks:
                recommendations.append(
                    f"URGENT: {len(coordinated_attacks)} coordinated attacks detected - "
                    "immediate investigation and response required"
                )
            
            review_bombing = [a for a in critical_issues 
                            if a.anomaly_type.value == 'review_bombing']
            if review_bombing:
                recommendations.append(
                    f"High volume of suspicious reviews detected - "
                    "consider temporary review restrictions"
                )
            
            rating_manipulation = [a for a in critical_issues 
                                 if a.anomaly_type.value in ['rating_manipulation', 'sudden_rating_drop']]
            if rating_manipulation:
                recommendations.append(
                    "Rating manipulation detected - review rating authenticity and "
                    "consider adjustments to app store metrics"
                )
        
        # Pattern-based recommendations
        spam_reviews = [a for a in all_anomalies if a.anomaly_type.value == 'spam_review']
        if len(spam_reviews) > 5:
            recommendations.append(
                f"Multiple spam reviews detected ({len(spam_reviews)}) - "
                "implement enhanced content filtering"
            )
        
        crash_spikes = [a for a in all_anomalies if a.anomaly_type.value == 'crash_report_spike']
        if crash_spikes:
            recommendations.append(
                "Crash report spike detected - investigate recent app updates and "
                "consider emergency patch if needed"
            )
        
        payment_issues = [a for a in all_anomalies if a.anomaly_type.value == 'payment_issue_spike']
        if payment_issues:
            recommendations.append(
                "Payment issue spike detected - review billing system and "
                "provide customer support escalation"
            )
        
        # General recommendations
        if len(all_anomalies) > 20:
            recommendations.append(
                f"High number of anomalies detected ({len(all_anomalies)}) - "
                "comprehensive review of app store presence recommended"
            )
        
        # Default recommendation if no specific patterns
        if not recommendations and all_anomalies:
            recommendations.append(
                "Anomalies detected - monitor trends and investigate outliers"
            )
        
        return recommendations
    
    def _update_statistics(self, result: AnomalyResult):
        """Update processing statistics"""
        self.processing_stats['total_reviews_processed'] += result.total_reviews_analyzed
        self.processing_stats['total_anomalies_detected'] += len(result.classified_anomalies)
        self.processing_stats['detection_runs'] += 1
        
        # Update average processing time
        current_avg = self.processing_stats['average_processing_time']
        new_avg = (current_avg * (self.processing_stats['detection_runs'] - 1) + result.processing_time) / self.processing_stats['detection_runs']
        self.processing_stats['average_processing_time'] = new_avg
    
    def configure_sensitivity(self, 
                            global_sensitivity: Optional[float] = None,
                            method_configs: Optional[Dict[str, Any]] = None):
        """
        Configure detection sensitivity and thresholds.
        
        Args:
            global_sensitivity: Global sensitivity multiplier
            method_configs: Method-specific configuration updates
        """
        if global_sensitivity is not None:
            self.classifier.configure_sensitivity(global_sensitivity=global_sensitivity)
        
        if method_configs:
            for method, config_updates in method_configs.items():
                if method in self.detectors:
                    detector = self.detectors[method]
                    
                    # Update detector configuration
                    if hasattr(detector, 'config'):
                        for key, value in config_updates.items():
                            if hasattr(detector.config, key):
                                setattr(detector.config, key, value)
                                logger.info(f"Updated {method} detector {key} to {value}")
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        stats = {
            'processing_stats': self.processing_stats.copy(),
            'pipeline_config': {
                'detectors_enabled': list(self.detectors.keys()),
                'min_reviews_for_detection': self.config.min_reviews_for_detection,
                'min_severity_level': self.config.min_severity_level,
                'max_anomalies_per_app': self.config.max_anomalies_per_app
            },
            'detector_stats': {}
        }
        
        # Get detector-specific statistics
        for method, detector in self.detectors.items():
            if hasattr(detector, 'get_statistics'):
                stats['detector_stats'][method] = detector.get_statistics()
        
        # Add classifier statistics
        stats['classifier_stats'] = self.classifier.get_statistics()
        
        # Add embedding pipeline statistics
        if self.embedding_pipeline:
            stats['embedding_stats'] = self.embedding_pipeline.get_pipeline_statistics()
        
        return stats
    
    def clear_caches(self):
        """Clear all caches in the pipeline"""
        if self.embedding_pipeline:
            self.embedding_pipeline.clear_caches()
        
        logger.info("Pipeline caches cleared")
    
    async def batch_detect_anomalies(self, 
                                    apps_reviews: Dict[str, List[Dict[str, Any]]]) -> Dict[str, AnomalyResult]:
        """
        Detect anomalies for multiple apps in batch.
        
        Args:
            apps_reviews: Dictionary mapping app_id to list of reviews
            
        Returns:
            Dictionary mapping app_id to anomaly detection results
        """
        logger.info(f"Starting batch anomaly detection for {len(apps_reviews)} apps")
        
        results = {}
        
        # Process apps sequentially (could be made concurrent if needed)
        for app_id, reviews in apps_reviews.items():
            try:
                result = await self.detect_anomalies(app_id, reviews)
                results[app_id] = result
                logger.info(f"Completed anomaly detection for {app_id}")
            except Exception as e:
                logger.error(f"Batch detection failed for {app_id}: {e}")
                results[app_id] = self._create_empty_result(app_id, len(reviews))
        
        logger.info(f"Batch anomaly detection complete: {len(results)} apps processed")
        return results