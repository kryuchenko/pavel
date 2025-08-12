"""
Preprocessing pipeline for Google Play reviews.

Orchestrates the complete preprocessing workflow:
1. Text normalization
2. Language detection
3. Sentence segmentation
4. Content deduplication
5. MongoDB integration
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

from pymongo import MongoClient
from pymongo.errors import BulkWriteError

from pavel.core.config import get_config
from pavel.core.logger import get_logger
from .normalizer import TextNormalizer, NormalizationStats
from .language_detector import LanguageDetector, LanguageDetectionResult
from .sentence_splitter import SentenceSplitter, SentenceSplitResult
from .deduplicator import ContentDeduplicator, DeduplicationResult
# Lazy import to avoid circular dependency
# from ..classification import get_complaint_classifier, ComplaintPrediction

logger = get_logger(__name__)

@dataclass
class ProcessedReview:
    """Structure for a fully processed review"""
    review_id: str
    app_id: str
    original_content: str
    normalized_content: str
    detected_language: str
    language_confidence: float
    sentences: List[str]
    sentence_count: int
    avg_sentence_length: float
    is_duplicate: bool
    duplicate_group_id: Optional[str]
    is_complaint: bool
    complaint_confidence: float
    processing_stats: Dict[str, Any]
    processed_at: datetime

@dataclass
class ProcessingBatchResult:
    """Result of processing a batch of reviews"""
    total_reviews: int
    processed_reviews: int
    failed_reviews: int
    duplicate_reviews: int
    unique_reviews: int
    complaint_reviews: int
    non_complaint_reviews: int
    language_distribution: Dict[str, int]
    processing_stats: Dict[str, Any]
    duration_seconds: float

class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for Google Play reviews.
    
    Features:
    - Configurable processing stages
    - Batch processing with error handling
    - MongoDB integration
    - Progress tracking and statistics
    - Resume capability for interrupted processing
    """
    
    def __init__(self, mongo_client: Optional[MongoClient] = None):
        self.config = get_config()
        self.mongo_client = mongo_client or self._get_mongo_client()
        self.db = self.mongo_client[self.config.MONGODB_DATABASE]
        self.reviews_collection = self.db.reviews
        
        # Initialize processing components
        self.normalizer = TextNormalizer(
            preserve_emojis=True,
            max_repeated_chars=3,
            remove_urls=True
        )
        
        self.language_detector = LanguageDetector(
            min_confidence=0.7,
            min_text_length=10,
            use_locale_fallback=True
        )
        
        self.sentence_splitter = SentenceSplitter(
            min_sentence_length=5,
            max_sentence_length=500,
            preserve_emojis=True
        )
        
        self.deduplicator = ContentDeduplicator(
            similarity_threshold=0.85,
            min_length_for_comparison=10,
            normalize_for_comparison=True
        )
        
        # Initialize complaint classifier (lazy loading)
        self.complaint_classifier = None
        
    def _get_mongo_client(self) -> MongoClient:
        """Get MongoDB client from config"""
        return MongoClient(self.config.MONGODB_URI)
        
    async def process_single_review(self, review_data: Dict) -> ProcessedReview:
        """
        Process a single review through the complete pipeline.
        
        Args:
            review_data: Review document from MongoDB
            
        Returns:
            ProcessedReview with all processing results
        """
        review_id = review_data.get('reviewId', '')
        app_id = review_data.get('appId', '')
        original_content = review_data.get('content', '')
        locale = review_data.get('locale')
        
        logger.debug(f"Processing review {review_id}: {original_content[:50]}...")
        
        processing_stats = {}
        start_time = datetime.now(timezone.utc)
        
        try:
            # Stage 1: Text normalization
            normalized_content, norm_stats = self.normalizer.normalize(original_content)
            processing_stats['normalization'] = asdict(norm_stats)
            
            # Stage 2: Language detection
            lang_result = self.language_detector.detect(normalized_content, locale)
            processing_stats['language_detection'] = {
                'detected_language': lang_result.language,
                'confidence': lang_result.confidence,
                'method': lang_result.method,
                'alternatives': lang_result.alternatives[:3]  # Top 3 alternatives
            }
            
            # Stage 3: Sentence segmentation
            sentence_result = self.sentence_splitter.split(normalized_content, lang_result.language)
            processing_stats['sentence_splitting'] = {
                'sentence_count': sentence_result.sentence_count,
                'avg_sentence_length': sentence_result.avg_sentence_length,
                'method': sentence_result.method
            }
            
            # Stage 4: Complaint classification (lazy loading)
            if self.complaint_classifier is None:
                from ..classification import get_complaint_classifier
                self.complaint_classifier = get_complaint_classifier()
                
            complaint_prediction = self.complaint_classifier.predict_single(normalized_content)
            processing_stats['complaint_classification'] = {
                'is_complaint': complaint_prediction.is_complaint,
                'confidence': complaint_prediction.confidence,
                'probability': complaint_prediction.complaint_probability
            }
            
            # Create processed review object
            processed_review = ProcessedReview(
                review_id=review_id,
                app_id=app_id,
                original_content=original_content,
                normalized_content=normalized_content,
                detected_language=lang_result.language,
                language_confidence=lang_result.confidence,
                sentences=sentence_result.sentences,
                sentence_count=sentence_result.sentence_count,
                avg_sentence_length=sentence_result.avg_sentence_length,
                is_duplicate=False,  # Will be set during deduplication
                duplicate_group_id=None,
                is_complaint=complaint_prediction.is_complaint,
                complaint_confidence=complaint_prediction.confidence,
                processing_stats=processing_stats,
                processed_at=datetime.now(timezone.utc)
            )
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            processing_stats['duration_seconds'] = duration
            
            logger.debug(f"Processed review {review_id} in {duration:.2f}s: "
                        f"{lang_result.language} ({lang_result.confidence:.2f}), "
                        f"{sentence_result.sentence_count} sentences")
            
            return processed_review
            
        except Exception as e:
            logger.error(f"Failed to process review {review_id}: {e}")
            # Return minimal processed review with error info
            return ProcessedReview(
                review_id=review_id,
                app_id=app_id,
                original_content=original_content,
                normalized_content="",
                detected_language="unknown",
                language_confidence=0.0,
                sentences=[],
                sentence_count=0,
                avg_sentence_length=0.0,
                is_duplicate=False,
                duplicate_group_id=None,
                processing_stats={'error': str(e)},
                processed_at=datetime.now(timezone.utc)
            )
            
    async def process_batch(self, 
                           app_id: str,
                           batch_size: int = 100,
                           skip_processed: bool = True) -> ProcessingBatchResult:
        """
        Process a batch of reviews for a specific app.
        
        Args:
            app_id: App identifier
            batch_size: Number of reviews to process in each batch
            skip_processed: Whether to skip already processed reviews
            
        Returns:
            ProcessingBatchResult with batch statistics
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info(f"Starting batch processing for app {app_id}, batch_size={batch_size}")
        
        # Query for unprocessed reviews
        query = {"appId": app_id}
        if skip_processed:
            query["processed"] = {"$ne": True}
            
        # Get reviews to process
        reviews_cursor = self.reviews_collection.find(query).limit(batch_size)
        reviews_data = list(reviews_cursor)
        
        if not reviews_data:
            logger.info(f"No reviews to process for app {app_id}")
            return ProcessingBatchResult(
                total_reviews=0,
                processed_reviews=0,
                failed_reviews=0,
                duplicate_reviews=0,
                unique_reviews=0,
                complaint_reviews=0,
                non_complaint_reviews=0,
                language_distribution={},
                processing_stats={},
                duration_seconds=0.0
            )
            
        logger.info(f"Processing {len(reviews_data)} reviews for app {app_id}")
        
        # Process individual reviews
        processed_reviews = []
        failed_count = 0
        
        for review_data in reviews_data:
            try:
                processed = await self.process_single_review(review_data)
                processed_reviews.append(processed)
            except Exception as e:
                failed_count += 1
                logger.error(f"Failed to process review {review_data.get('reviewId', 'unknown')}: {e}")
                
        # Stage 4: Deduplication (on the batch)
        if processed_reviews:
            logger.info(f"Running deduplication on {len(processed_reviews)} processed reviews")
            
            # Extract normalized content for deduplication
            texts = [review.normalized_content for review in processed_reviews]
            dedup_result = self.deduplicator.deduplicate(texts, method="both")
            
            # Update processed reviews with deduplication info
            self._apply_deduplication_results(processed_reviews, dedup_result)
            
        # Update MongoDB with processed results
        updated_count = await self._save_processed_reviews(processed_reviews)
        
        # Calculate statistics
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        language_dist = {}
        for review in processed_reviews:
            lang = review.detected_language
            language_dist[lang] = language_dist.get(lang, 0) + 1
            
        processing_stats = {
            "avg_normalization_ratio": self._calculate_avg_normalization_ratio(processed_reviews),
            "avg_language_confidence": self._calculate_avg_language_confidence(processed_reviews),
            "avg_sentences_per_review": self._calculate_avg_sentences(processed_reviews),
            "deduplication": {
                "similarity_threshold": self.deduplicator.similarity_threshold,
                "duplicate_groups": len(dedup_result.duplicate_groups) if processed_reviews else 0,
                "duplication_rate": dedup_result.duplicate_count / len(processed_reviews) if processed_reviews else 0
            }
        }
        
        result = ProcessingBatchResult(
            total_reviews=len(reviews_data),
            processed_reviews=len(processed_reviews),
            failed_reviews=failed_count,
            duplicate_reviews=sum(1 for r in processed_reviews if r.is_duplicate),
            unique_reviews=sum(1 for r in processed_reviews if not r.is_duplicate),
            complaint_reviews=sum(1 for r in processed_reviews if r.is_complaint),
            non_complaint_reviews=sum(1 for r in processed_reviews if not r.is_complaint),
            language_distribution=language_dist,
            processing_stats=processing_stats,
            duration_seconds=duration
        )
        
        logger.info(f"Batch processing complete for {app_id}: "
                   f"{result.processed_reviews} processed, {result.failed_reviews} failed, "
                   f"{result.duplicate_reviews} duplicates in {duration:.1f}s")
        
        return result
        
    def _apply_deduplication_results(self, 
                                   processed_reviews: List[ProcessedReview],
                                   dedup_result: DeduplicationResult):
        """Apply deduplication results to processed reviews"""
        
        # Create mapping of text to group info
        text_to_group = {}
        
        for group in dedup_result.duplicate_groups:
            for i, text in enumerate(group.duplicate_texts):
                is_canonical = (i == 0)  # First text is canonical
                text_to_group[text] = {
                    'group_id': group.group_id,
                    'is_duplicate': not is_canonical,
                    'similarity_score': group.similarity_scores[i] if i < len(group.similarity_scores) else 1.0
                }
                
        # Apply to processed reviews
        for review in processed_reviews:
            group_info = text_to_group.get(review.normalized_content)
            if group_info:
                review.is_duplicate = group_info['is_duplicate']
                review.duplicate_group_id = group_info['group_id']
                review.processing_stats['deduplication'] = {
                    'group_id': group_info['group_id'],
                    'similarity_score': group_info['similarity_score']
                }
                
    async def _save_processed_reviews(self, processed_reviews: List[ProcessedReview]) -> int:
        """Save processed reviews back to MongoDB"""
        
        if not processed_reviews:
            return 0
            
        # Prepare updates for MongoDB
        updates = []
        
        for review in processed_reviews:
            # Create update document
            update_doc = {
                "$set": {
                    "processed": True,
                    "processedAt": review.processed_at,
                    "normalizedContent": review.normalized_content,
                    "detectedLanguage": review.detected_language,
                    "languageConfidence": review.language_confidence,
                    "sentences": [
                        {
                            "text": sentence,
                            "index": i,
                            "length": len(sentence),
                            "language": review.detected_language
                        }
                        for i, sentence in enumerate(review.sentences)
                    ],
                    "sentenceCount": review.sentence_count,
                    "avgSentenceLength": review.avg_sentence_length,
                    "isDuplicate": review.is_duplicate,
                    "duplicateGroupId": review.duplicate_group_id,
                    "processingStats": review.processing_stats,
                    "updatedAt": datetime.now(timezone.utc)
                }
            }
            
            updates.append({
                "filter": {"_id": f"{review.app_id}:{review.review_id}"},
                "update": update_doc,
                "upsert": False
            })
            
        # Perform bulk update
        try:
            operations = [
                {"updateOne": update} for update in updates
            ]
            
            result = self.reviews_collection.bulk_write(operations, ordered=False)
            updated_count = result.modified_count
            
            logger.info(f"Updated {updated_count} reviews in MongoDB")
            return updated_count
            
        except BulkWriteError as e:
            logger.error(f"Bulk write error: {e.details}")
            # Count successful updates
            return len(updates) - len(e.details.get('writeErrors', []))
        except Exception as e:
            logger.error(f"Failed to save processed reviews: {e}")
            return 0
            
    def _calculate_avg_normalization_ratio(self, reviews: List[ProcessedReview]) -> float:
        """Calculate average normalization ratio (normalized/original length)"""
        ratios = []
        for review in reviews:
            if review.original_content and 'normalization' in review.processing_stats:
                original_len = review.processing_stats['normalization'].get('original_length', 0)
                normalized_len = review.processing_stats['normalization'].get('normalized_length', 0)
                if original_len > 0:
                    ratios.append(normalized_len / original_len)
        return sum(ratios) / len(ratios) if ratios else 0.0
        
    def _calculate_avg_language_confidence(self, reviews: List[ProcessedReview]) -> float:
        """Calculate average language detection confidence"""
        confidences = [review.language_confidence for review in reviews]
        return sum(confidences) / len(confidences) if confidences else 0.0
        
    def _calculate_avg_sentences(self, reviews: List[ProcessedReview]) -> float:
        """Calculate average sentences per review"""
        sentence_counts = [review.sentence_count for review in reviews]
        return sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0.0
        
    async def process_app_complete(self, 
                                  app_id: str,
                                  batch_size: int = 100) -> List[ProcessingBatchResult]:
        """
        Process all unprocessed reviews for an app.
        
        Args:
            app_id: App identifier
            batch_size: Batch size for processing
            
        Returns:
            List of ProcessingBatchResults for each batch
        """
        logger.info(f"Starting complete processing for app {app_id}")
        
        results = []
        
        while True:
            # Process next batch
            batch_result = await self.process_batch(app_id, batch_size, skip_processed=True)
            
            if batch_result.total_reviews == 0:
                # No more reviews to process
                break
                
            results.append(batch_result)
            
            # Log progress
            total_processed = sum(r.processed_reviews for r in results)
            total_failed = sum(r.failed_reviews for r in results)
            
            logger.info(f"App {app_id} progress: {total_processed} processed, {total_failed} failed")
            
        logger.info(f"Complete processing finished for app {app_id}: {len(results)} batches")
        return results
        
    def get_processing_status(self, app_id: str) -> Dict:
        """Get processing status for an app"""
        
        total_reviews = self.reviews_collection.count_documents({"appId": app_id})
        processed_reviews = self.reviews_collection.count_documents({
            "appId": app_id,
            "processed": True
        })
        
        # Get sample of processed reviews for stats
        sample_processed = list(self.reviews_collection.find({
            "appId": app_id,
            "processed": True
        }).limit(100))
        
        # Language distribution
        lang_dist = {}
        for review in sample_processed:
            lang = review.get('detectedLanguage', 'unknown')
            lang_dist[lang] = lang_dist.get(lang, 0) + 1
            
        return {
            "app_id": app_id,
            "total_reviews": total_reviews,
            "processed_reviews": processed_reviews,
            "unprocessed_reviews": total_reviews - processed_reviews,
            "processing_progress": processed_reviews / total_reviews if total_reviews > 0 else 0,
            "sample_language_distribution": lang_dist
        }
        
    def close(self):
        """Clean up resources"""
        if self.mongo_client:
            self.mongo_client.close()