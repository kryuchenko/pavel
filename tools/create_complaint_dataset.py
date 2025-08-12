#!/usr/bin/env python3
"""
Create labeled dataset for complaint/non-complaint classification

This script extracts inDriver reviews and creates a balanced dataset
for training a local complaint classifier.
"""

import sys
import os
import json
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pymongo
from pavel.core.config import get_config
from pavel.core.logger import get_logger

logger = get_logger(__name__)

class ComplaintDatasetCreator:
    def __init__(self):
        self.config = get_config()
        self.client = pymongo.MongoClient(self.config.DB_URI)
        self.db = self.client[self.config.DB_NAME]
        self.reviews_collection = self.db.reviews
        
    def extract_reviews_sample(self, sample_size: int = 1000) -> List[Dict[str, Any]]:
        """Extract a balanced sample of reviews for labeling"""
        logger.info(f"Extracting {sample_size} reviews for dataset creation")
        
        # Get balanced sample by rating
        pipeline = [
            {"$match": {"appId": "sinet.startup.inDriver"}},
            {"$sample": {"size": sample_size}},
            {"$project": {
                "reviewId": "$reviewId",
                "content": "$content",
                "score": "$score",
                "locale": "$locale",
                "at": "$at",
                "appVersion": "$appVersion"
            }}
        ]
        
        try:
            reviews = list(self.reviews_collection.aggregate(pipeline))
            logger.info(f"✅ Extracted {len(reviews)} reviews")
            return reviews
        except Exception as e:
            logger.error(f"Error extracting reviews: {e}")
            # Fallback to simple find
            reviews = list(self.reviews_collection.find(
                {"appId": "sinet.startup.inDriver"},
                {
                    "reviewId": 1,
                    "content": 1, 
                    "score": 1,
                    "locale": 1,
                    "at": 1,
                    "appVersion": 1
                }
            ).limit(sample_size))
            logger.info(f"✅ Extracted {len(reviews)} reviews (fallback)")
            return reviews
    
    def create_prelabeled_dataset(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create pre-labeled dataset using heuristic rules"""
        logger.info("Creating pre-labeled dataset using heuristics")
        
        # Complaint indicators (multilingual)
        complaint_keywords = {
            'en': ['crash', 'bug', 'error', 'problem', 'issue', 'broken', 'fix', 'worst', 'hate', 'terrible', 'awful', 'useless', 'doesn\'t work', 'not working'],
            'ru': ['крэш', 'баг', 'ошибка', 'проблема', 'сломан', 'исправь', 'не работает', 'ужасно', 'плохо', 'кошмар', 'глючит', 'тормозит'],
            'es': ['error', 'problema', 'fallo', 'mal', 'roto', 'pésimo', 'horrible', 'no funciona', 'terrible']
        }
        
        # Positive indicators  
        positive_keywords = {
            'en': ['great', 'awesome', 'excellent', 'perfect', 'love', 'best', 'amazing', 'fantastic', 'good', 'nice', 'helpful'],
            'ru': ['отлично', 'супер', 'классно', 'круто', 'люблю', 'лучший', 'замечательно', 'хорошо', 'нравится'],
            'es': ['excelente', 'genial', 'perfecto', 'increíble', 'bueno', 'fantástico', 'maravilloso']
        }
        
        labeled_data = []
        
        for review in reviews:
            content = review.get('content') or ''
            if not content:
                continue
            content = content.lower()
            score = review.get('score', 5)
            locale = review.get('locale', 'en_US')
            lang = locale.split('_')[0]  # Extract language code
            
            # Rule-based labeling
            is_complaint = False
            confidence = 0.0
            
            # Low rating usually indicates complaints
            if score <= 2:
                is_complaint = True
                confidence += 0.6
                
            # High rating usually indicates satisfaction  
            elif score >= 4:
                is_complaint = False
                confidence += 0.4
            
            # Check for complaint keywords
            complaint_words = complaint_keywords.get(lang, complaint_keywords['en'])
            for keyword in complaint_words:
                if keyword in content:
                    is_complaint = True
                    confidence += 0.3
                    break
                    
            # Check for positive keywords
            positive_words = positive_keywords.get(lang, positive_keywords['en'])
            for keyword in positive_words:
                if keyword in content:
                    is_complaint = False
                    confidence += 0.2
                    break
            
            # Only include high-confidence labels
            if confidence >= 0.6:
                labeled_data.append({
                    'reviewId': review['reviewId'],
                    'content': review['content'],
                    'score': score,
                    'locale': locale,
                    'language': lang,
                    'at': review.get('at'),
                    'appVersion': review.get('appVersion'),
                    'label': 1 if is_complaint else 0,  # 1 = complaint, 0 = not complaint
                    'confidence': min(confidence, 1.0),
                    'labeling_method': 'heuristic'
                })
        
        logger.info(f"✅ Created {len(labeled_data)} high-confidence labels")
        return labeled_data
    
    def balance_dataset(self, labeled_data: List[Dict[str, Any]], max_per_class: int = 500) -> List[Dict[str, Any]]:
        """Balance the dataset between complaint/non-complaint"""
        complaints = [item for item in labeled_data if item['label'] == 1]
        non_complaints = [item for item in labeled_data if item['label'] == 0]
        
        logger.info(f"Before balancing: {len(complaints)} complaints, {len(non_complaints)} non-complaints")
        
        # Randomly sample to balance classes
        random.shuffle(complaints)
        random.shuffle(non_complaints)
        
        balanced_complaints = complaints[:max_per_class]
        balanced_non_complaints = non_complaints[:max_per_class]
        
        balanced_dataset = balanced_complaints + balanced_non_complaints
        random.shuffle(balanced_dataset)
        
        logger.info(f"After balancing: {len(balanced_complaints)} complaints, {len(balanced_non_complaints)} non-complaints")
        return balanced_dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str):
        """Save dataset to JSON file"""
        output_path = Path(__file__).parent.parent / "data" / filename
        output_path.parent.mkdir(exist_ok=True)
        
        # Add metadata
        dataset_info = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_samples': len(dataset),
                'complaints': len([item for item in dataset if item['label'] == 1]),
                'non_complaints': len([item for item in dataset if item['label'] == 0]),
                'languages': list(set(item['language'] for item in dataset)),
                'description': 'inDriver review complaint classification dataset',
                'label_encoding': {'0': 'not_complaint', '1': 'complaint'}
            },
            'data': dataset
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ Dataset saved to {output_path}")
        return output_path

def main():
    """Create complaint classification dataset"""
    print("🏗️  Creating inDriver Complaint Classification Dataset")
    print("=" * 60)
    
    creator = ComplaintDatasetCreator()
    
    # Step 1: Extract reviews sample
    print("📥 Step 1: Extracting reviews sample...")
    reviews = creator.extract_reviews_sample(sample_size=1500)
    
    if not reviews:
        print("❌ No reviews found!")
        return
    
    # Step 2: Create pre-labeled dataset
    print("🏷️  Step 2: Creating pre-labeled dataset...")
    labeled_data = creator.create_prelabeled_dataset(reviews)
    
    if not labeled_data:
        print("❌ No high-confidence labels created!")
        return
        
    # Step 3: Balance the dataset
    print("⚖️  Step 3: Balancing dataset...")
    balanced_dataset = creator.balance_dataset(labeled_data, max_per_class=400)
    
    # Step 4: Save dataset
    print("💾 Step 4: Saving dataset...")
    output_path = creator.save_dataset(balanced_dataset, "complaint_classification_dataset.json")
    
    # Summary
    print("\n🎯 Dataset Creation Summary:")
    print(f"   📊 Total samples: {len(balanced_dataset)}")
    print(f"   🚨 Complaints: {len([item for item in balanced_dataset if item['label'] == 1])}")
    print(f"   ✅ Non-complaints: {len([item for item in balanced_dataset if item['label'] == 0])}")
    print(f"   🌍 Languages: {', '.join(set(item['language'] for item in balanced_dataset))}")
    print(f"   💾 Saved to: {output_path}")
    print(f"\n🎉 Ready for training!")

if __name__ == "__main__":
    main()