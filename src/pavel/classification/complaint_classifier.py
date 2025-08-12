"""
Complaint Classification Module for Stage 3

Integrates trained ML model for complaint/non-complaint classification
using multilingual embeddings.
"""

import os
import numpy as np
import joblib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from pavel.core.logger import get_logger
from pavel.embeddings.embedding_generator import EmbeddingGenerator, EmbeddingConfig

logger = get_logger(__name__)

@dataclass
class ComplaintPrediction:
    """Result of complaint classification"""
    text: str
    is_complaint: bool
    confidence: float
    complaint_probability: float

class ComplaintClassifier:
    """
    Production complaint classifier using trained ML model
    
    Integrates with Stage 3 preprocessing to filter complaint reviews
    before they proceed to embeddings and anomaly detection stages.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or self._get_default_model_path()
        self.model = None
        self.scaler = None
        self.metadata = None
        self.embedding_generator = None
        self._is_loaded = False
        
    def _get_default_model_path(self) -> str:
        """Get default model path"""
        base_path = Path(__file__).parent.parent.parent.parent
        return str(base_path / "models" / "complaint_classifier.joblib")
        
    def load_model(self):
        """Load the trained model and components"""
        if self._is_loaded:
            return
            
        logger.info(f"Loading complaint classifier from {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
            
        try:
            # Load model components
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.metadata = model_data['metadata']
            
            # Initialize embedding generator
            embedding_config = EmbeddingConfig(
                model_name=self.metadata['embedding_model'],
                batch_size=32
            )
            self.embedding_generator = EmbeddingGenerator(embedding_config)
            
            self._is_loaded = True
            logger.info("✅ Complaint classifier loaded successfully")
            logger.info(f"   Model: {self.metadata['model_type']}")
            logger.info(f"   Trained: {self.metadata.get('trained_at', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def predict_single(self, text: str, threshold: float = 0.5) -> ComplaintPrediction:
        """Predict if a single text is a complaint"""
        predictions = self.predict_batch([text], threshold)
        return predictions[0]
        
    def predict_batch(self, texts: List[str], threshold: float = 0.5) -> List[ComplaintPrediction]:
        """Predict complaints for a batch of texts"""
        if not self._is_loaded:
            self.load_model()
            
        if not texts:
            return []
            
        logger.debug(f"Classifying {len(texts)} texts with threshold={threshold}")
        
        try:
            # Generate embeddings
            embeddings = self._generate_embeddings(texts)
            
            # Scale features
            embeddings_scaled = self.scaler.transform(embeddings)
            
            # Predict probabilities
            probabilities = self.model.predict_proba(embeddings_scaled)
            complaint_probs = probabilities[:, 1]  # Probability of being complaint
            
            # Create predictions
            predictions = []
            for text, prob in zip(texts, complaint_probs):
                is_complaint = bool(prob >= threshold)  # Convert numpy bool to Python bool
                confidence = float(prob if is_complaint else (1 - prob))
                
                predictions.append(ComplaintPrediction(
                    text=text,
                    is_complaint=is_complaint,
                    confidence=confidence,
                    complaint_probability=float(prob)
                ))
                
            logger.debug(f"✅ Classified {len(predictions)} texts: "
                        f"{sum(p.is_complaint for p in predictions)} complaints, "
                        f"{sum(not p.is_complaint for p in predictions)} non-complaints")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return default predictions on error
            return [ComplaintPrediction(
                text=text,
                is_complaint=True,  # Conservative: assume complaint on error
                confidence=0.5,
                complaint_probability=0.5
            ) for text in texts]
            
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts using the same model as training"""
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_generator.generate_batch(batch)
            
            for emb_result in batch_embeddings:
                if emb_result.embedding is not None:
                    embeddings.append(emb_result.embedding)
                else:
                    # Use zero vector for failed embeddings
                    embeddings.append(np.zeros(384))  # E5-small dimension
                    
        return np.array(embeddings)
        
    def filter_complaints(self, reviews: List[Dict[str, Any]], 
                         threshold: float = 0.5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Filter reviews to keep only complaints
        
        Returns:
            - filtered_reviews: Only complaint reviews
            - stats: Filtering statistics
        """
        if not reviews:
            return [], {'total': 0, 'complaints': 0, 'non_complaints': 0}
            
        # Extract text content
        texts = [review.get('content', '') for review in reviews]
        
        # Predict complaints
        predictions = self.predict_batch(texts, threshold)
        
        # Filter reviews
        filtered_reviews = []
        for review, prediction in zip(reviews, predictions):
            if prediction.is_complaint:
                # Add classification metadata to review
                review['complaint_classification'] = {
                    'is_complaint': True,
                    'confidence': prediction.confidence,
                    'probability': prediction.complaint_probability,
                    'classified_at': self.metadata.get('trained_at'),
                    'model_type': self.metadata.get('model_type')
                }
                filtered_reviews.append(review)
        
        # Generate stats
        stats = {
            'total': len(reviews),
            'complaints': len(filtered_reviews),
            'non_complaints': len(reviews) - len(filtered_reviews),
            'complaint_rate': len(filtered_reviews) / len(reviews) if reviews else 0.0,
            'threshold': threshold
        }
        
        logger.info(f"🚨 Complaint filtering: {stats['complaints']}/{stats['total']} "
                   f"({100*stats['complaint_rate']:.1f}%) reviews classified as complaints")
        
        return filtered_reviews, stats

def get_complaint_classifier() -> ComplaintClassifier:
    """Get singleton complaint classifier instance"""
    if not hasattr(get_complaint_classifier, '_instance'):
        get_complaint_classifier._instance = ComplaintClassifier()
    return get_complaint_classifier._instance