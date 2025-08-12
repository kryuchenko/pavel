#!/usr/bin/env python3
"""
Train complaint classification model using multilingual embeddings

This creates a local ML model for Stage 4 complaint filtering using the
existing E5-multilingual embeddings infrastructure.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pavel.core.logger import get_logger
from pavel.embeddings.embedding_generator import EmbeddingGenerator, EmbeddingConfig

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

logger = get_logger(__name__)

class ComplaintClassifierTrainer:
    def __init__(self):
        # Initialize embedding generator  
        embedding_config = EmbeddingConfig(
            model_name="intfloat/multilingual-e5-small",
            device="cpu",
            batch_size=32
        )
        self.embedding_generator = EmbeddingGenerator(embedding_config)
        self.scaler = StandardScaler()
        
    def load_dataset(self, dataset_path: str) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load dataset from JSON file"""
        logger.info(f"Loading dataset from {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item['content'] for item in data['data']]
        labels = [item['label'] for item in data['data']]
        metadata = data['metadata']
        
        logger.info(f"✅ Loaded {len(texts)} samples")
        logger.info(f"   Complaints: {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
        logger.info(f"   Non-complaints: {len(labels) - sum(labels)} ({100*(len(labels) - sum(labels))/len(labels):.1f}%)")
        
        return texts, labels, metadata
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text data"""
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        try:
            # Use the embedding generator in batch mode
            embeddings = []
            batch_size = 32
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embedding_generator.generate_batch(batch)
                
                # Extract embedding vectors
                for emb_result in batch_embeddings:
                    if emb_result.embedding is not None:
                        embeddings.append(emb_result.embedding)
                    else:
                        # Use zero vector for failed embeddings
                        logger.warning(f"Failed to generate embedding, using zero vector")
                        embeddings.append(np.zeros(384))  # E5-small dimension
            
            embeddings_array = np.array(embeddings)
            logger.info(f"✅ Generated {embeddings_array.shape[0]} embeddings of dimension {embeddings_array.shape[1]}")
            return embeddings_array
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
            
    def augment_dataset(self, texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """Augment dataset to balance classes"""
        logger.info("Augmenting dataset for better balance...")
        
        complaints = [text for text, label in zip(texts, labels) if label == 1]
        non_complaints = [text for text, label in zip(texts, labels) if label == 0]
        
        logger.info(f"Original: {len(complaints)} complaints, {len(non_complaints)} non-complaints")
        
        # Create synthetic non-complaints by using high-rating reviews
        # This is a simple approach - in production we might use more sophisticated methods
        synthetic_non_complaints = [
            "Great app, works perfectly!",
            "Love this service, very convenient.",
            "Excellent experience, highly recommend.",
            "Perfect ride every time.",
            "Best taxi app I've used.",
            "Quick and reliable service.",
            "Amazing drivers and clean cars.",
            "Smooth booking process.",
            "Fair prices and good quality.",
            "Outstanding customer support.",
            
            # Russian
            "Отличное приложение, все работает!",
            "Очень удобно пользоваться.",
            "Превосходный сервис.",
            "Быстро и надежно.",
            "Лучшее приложение для такси.",
            "Хорошие водители.",
            "Справедливые цены.",
            "Отличная поддержка.",
            
            # Spanish  
            "Excelente aplicación, muy útil.",
            "Servicio perfecto y rápido.",
            "Los mejores conductores.",
            "Muy recomendable.",
            "Precios justos.",
            "Aplicación fácil de usar."
        ]
        
        # Balance the dataset
        target_per_class = min(400, len(complaints))
        
        # Add synthetic data to balance
        balanced_texts = complaints[:target_per_class]
        balanced_labels = [1] * len(balanced_texts)
        
        # Add original non-complaints + synthetic
        all_non_complaints = non_complaints + synthetic_non_complaints
        balanced_texts.extend(all_non_complaints[:target_per_class])
        balanced_labels.extend([0] * len(all_non_complaints[:target_per_class]))
        
        logger.info(f"Augmented: {sum(balanced_labels)} complaints, {len(balanced_labels) - sum(balanced_labels)} non-complaints")
        return balanced_texts, balanced_labels
        
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train multiple ML models and compare performance"""
        logger.info("Training multiple models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1)
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            if name == 'lightgbm':
                model.fit(X_train, y_train)  # LightGBM doesn't need scaling
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)  
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Evaluate
            accuracy = (y_pred == y_test).mean()
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled if name != 'lightgbm' else X_train, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            logger.info(f"✅ {name}: Accuracy={accuracy:.3f}, AUC={auc_score:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        # Select best model based on AUC
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
        logger.info(f"🏆 Best model: {best_model_name}")
        
        return results, best_model_name
        
    def save_model(self, model: Any, scaler: Any, metadata: Dict[str, Any], model_name: str = "complaint_classifier"):
        """Save trained model and preprocessing components"""
        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(exist_ok=True)
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'metadata': {
                **metadata,
                'trained_at': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'embedding_model': 'intfloat/multilingual-e5-small',
                'embedding_dimension': 384
            }
        }
        
        model_path = models_dir / f"{model_name}.joblib"
        joblib.dump(model_data, model_path)
        
        logger.info(f"✅ Model saved to {model_path}")
        return model_path

def main():
    """Train complaint classifier"""
    print("🤖 Training inDriver Complaint Classifier")
    print("=" * 50)
    
    trainer = ComplaintClassifierTrainer()
    
    # Step 1: Load dataset
    print("📥 Step 1: Loading dataset...")
    dataset_path = Path(__file__).parent.parent / "data" / "complaint_classification_dataset.json"
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please run create_complaint_dataset.py first")
        return
        
    texts, labels, metadata = trainer.load_dataset(dataset_path)
    
    # Step 2: Augment dataset for balance
    print("🔄 Step 2: Augmenting dataset...")
    texts, labels = trainer.augment_dataset(texts, labels)
    
    # Step 3: Generate embeddings
    print("🧠 Step 3: Generating embeddings...")
    X = trainer.generate_embeddings(texts)
    y = np.array(labels)
    
    # Step 4: Train models
    print("🏋️  Step 4: Training models...")
    results, best_model_name = trainer.train_models(X, y)
    
    # Step 5: Save best model
    print("💾 Step 5: Saving best model...")
    best_model = results[best_model_name]['model']
    model_path = trainer.save_model(best_model, trainer.scaler, metadata)
    
    # Summary
    print("\n🎯 Training Summary:")
    print(f"   📊 Dataset size: {len(texts)} samples")
    print(f"   🧠 Embeddings: {X.shape[1]}D E5-multilingual")
    print(f"   🏆 Best model: {best_model_name}")
    print(f"   🎯 Best accuracy: {results[best_model_name]['accuracy']:.3f}")
    print(f"   📈 Best AUC: {results[best_model_name]['auc']:.3f}")
    print(f"   💾 Saved to: {model_path}")
    
    print("\n📋 Model Performance:")
    for name, result in results.items():
        marker = "🏆" if name == best_model_name else "  "
        print(f"{marker} {name}: Acc={result['accuracy']:.3f}, AUC={result['auc']:.3f}")
    
    print(f"\n🎉 Complaint classifier ready for Stage 4 integration!")

if __name__ == "__main__":
    main()