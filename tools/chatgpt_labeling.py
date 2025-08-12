#!/usr/bin/env python3
"""
Professional complaint labeling using ChatGPT API

This script takes all inDriver reviews from MongoDB and labels each one
using ChatGPT-4 API to determine if it's a complaint or not.
"""

import sys
import os
import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import aiohttp
import pymongo
from dataclasses import dataclass

# Load .env.local if it exists (overrides .env)
from dotenv import load_dotenv
env_local = Path(__file__).parent.parent / ".env.local"
if env_local.exists():
    load_dotenv(env_local, override=True)
else:
    load_dotenv()  # Load regular .env

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pavel.core.config import get_config
from pavel.core.logger import get_logger

logger = get_logger(__name__)

@dataclass
class LabelingResult:
    """Result of ChatGPT labeling"""
    review_id: str
    content: str
    is_complaint: bool
    confidence: str  # "high", "medium", "low"
    reasoning: str
    processing_time: float
    api_cost: float

class ChatGPTLabeler:
    """Professional complaint labeling using ChatGPT API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key or self.api_key == 'YOUR_OPENAI_API_KEY_HERE':
            raise ValueError("OPENAI_API_KEY not found. Set it in .env.local or environment")
            
        self.config = get_config()
        self.client = pymongo.MongoClient(self.config.DB_URI)
        self.db = self.client[self.config.DB_NAME]
        self.reviews_collection = self.db.reviews
        
        # API settings
        self.model = "gpt-4o-mini"  # Use GPT-4o-mini for cost efficiency
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.requests_per_minute = 10000  # GPT-4o-mini higher limit
        self.batch_delay = 60.0 / self.requests_per_minute
        
        # Cost tracking (GPT-4o-mini pricing: $0.00015/1K input tokens, $0.0006/1K output tokens)
        self.total_cost = 0.0
        
    async def get_all_reviews(self) -> List[Dict[str, Any]]:
        """Get all inDriver reviews from MongoDB"""
        logger.info("Fetching all inDriver reviews from MongoDB...")
        
        query = {"appId": "sinet.startup.inDriver"}
        projection = {
            "reviewId": 1,
            "content": 1,
            "score": 1,
            "locale": 1,
            "at": 1,
            "appVersion": 1
        }
        
        reviews = list(self.reviews_collection.find(query, projection))
        logger.info(f"✅ Found {len(reviews)} inDriver reviews")
        return reviews
        
    def create_labeling_prompt(self, review_content: str, score: int, locale: str) -> str:
        """Create professional prompt for ChatGPT labeling"""
        return f"""You are an expert at analyzing mobile app reviews. Your task is to determine if a review represents a complaint or not.

REVIEW TO ANALYZE:
Content: "{review_content}"
Rating: {score}/5 stars
Locale: {locale}

DEFINITION:
- COMPLAINT: User expresses dissatisfaction, reports problems, bugs, issues, bad service, or negative experience
- NOT COMPLAINT: User expresses satisfaction, gives neutral feedback, asks questions, or provides positive comments

INSTRUCTIONS:
1. Analyze the semantic meaning, not just keywords
2. Consider cultural context and language nuances
3. Rating can be misleading - focus on content
4. Be strict: when in doubt, classify as NOT COMPLAINT

Respond ONLY with valid JSON:
{{
    "is_complaint": true/false,
    "confidence": "high/medium/low",
    "reasoning": "Brief explanation in English (max 50 words)"
}}"""

    async def label_single_review(self, session: aiohttp.ClientSession, 
                                 review: Dict[str, Any]) -> LabelingResult:
        """Label a single review using ChatGPT API"""
        start_time = time.time()
        
        content = review.get('content', '')
        if not content or len(content.strip()) < 5:
            # Skip empty or very short reviews
            return LabelingResult(
                review_id=review['reviewId'],
                content=content,
                is_complaint=False,
                confidence="low",
                reasoning="Review too short or empty",
                processing_time=0.001,
                api_cost=0.0
            )
        
        prompt = self.create_labeling_prompt(
            review_content=content,
            score=review.get('score', 5),
            locale=review.get('locale', 'unknown')
        )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.1,  # Low temperature for consistent results
            "response_format": {"type": "json_object"}
        }
        
        try:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API error for review {review['reviewId']}: {response.status} - {error_text}")
                    raise Exception(f"API request failed: {response.status}")
                
                data = await response.json()
                
                # Parse response
                result_text = data['choices'][0]['message']['content']
                result_json = json.loads(result_text)
                
                # Calculate cost (approximate)
                usage = data.get('usage', {})
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                cost = (input_tokens * 0.00015 + output_tokens * 0.0006) / 1000.0
                self.total_cost += cost
                
                processing_time = time.time() - start_time
                
                return LabelingResult(
                    review_id=review['reviewId'],
                    content=content,
                    is_complaint=result_json['is_complaint'],
                    confidence=result_json['confidence'],
                    reasoning=result_json['reasoning'],
                    processing_time=processing_time,
                    api_cost=cost
                )
                
        except Exception as e:
            logger.error(f"Failed to label review {review['reviewId']}: {e}")
            processing_time = time.time() - start_time
            
            # Return conservative fallback
            return LabelingResult(
                review_id=review['reviewId'],
                content=content,
                is_complaint=False,  # Conservative: assume not complaint on error
                confidence="low",
                reasoning=f"API error: {str(e)[:50]}",
                processing_time=processing_time,
                api_cost=0.0
            )
    
    async def label_batch_reviews(self, reviews: List[Dict[str, Any]], 
                                 batch_size: int = 50) -> List[LabelingResult]:
        """Label reviews in batches with rate limiting"""
        results = []
        total_reviews = len(reviews)
        
        logger.info(f"Starting batch labeling of {total_reviews} reviews...")
        logger.info(f"Batch size: {batch_size}, Rate limit: {self.requests_per_minute}/min")
        
        async with aiohttp.ClientSession() as session:
            for i in range(0, total_reviews, batch_size):
                batch = reviews[i:i + batch_size]
                batch_start = time.time()
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(total_reviews + batch_size - 1)//batch_size}: "
                          f"reviews {i+1}-{min(i+batch_size, total_reviews)}")
                
                # Process batch with rate limiting
                batch_results = []
                for j, review in enumerate(batch):
                    try:
                        result = await self.label_single_review(session, review)
                        batch_results.append(result)
                        
                        # Rate limiting delay
                        if j < len(batch) - 1:  # Don't delay after last item in batch
                            await asyncio.sleep(self.batch_delay)
                            
                    except Exception as e:
                        logger.error(f"Error processing review {review.get('reviewId', 'unknown')}: {e}")
                        # Add error result
                        batch_results.append(LabelingResult(
                            review_id=review.get('reviewId', 'unknown'),
                            content=review.get('content', ''),
                            is_complaint=False,
                            confidence="low", 
                            reasoning=f"Processing error: {str(e)[:30]}",
                            processing_time=0.0,
                            api_cost=0.0
                        ))
                
                results.extend(batch_results)
                
                batch_time = time.time() - batch_start
                complaints = sum(1 for r in batch_results if r.is_complaint)
                avg_confidence = sum(1 for r in batch_results if r.confidence == "high") / len(batch_results)
                
                logger.info(f"✅ Batch completed in {batch_time:.1f}s: "
                          f"{complaints}/{len(batch_results)} complaints, "
                          f"{avg_confidence*100:.1f}% high confidence")
                logger.info(f"💰 Running cost: ${self.total_cost:.3f}")
                
        return results
    
    def save_labeled_dataset(self, results: List[LabelingResult], 
                           output_file: str = "chatgpt_labeled_dataset.json"):
        """Save labeled dataset to JSON file"""
        output_path = Path(__file__).parent.parent / "data" / output_file
        output_path.parent.mkdir(exist_ok=True)
        
        # Convert results to serializable format
        dataset_data = []
        for result in results:
            dataset_data.append({
                'reviewId': result.review_id,
                'content': result.content,
                'is_complaint': result.is_complaint,
                'confidence': result.confidence,
                'reasoning': result.reasoning,
                'processing_time': result.processing_time,
                'api_cost': result.api_cost
            })
        
        # Add metadata
        complaints_count = sum(1 for r in results if r.is_complaint)
        high_conf_count = sum(1 for r in results if r.confidence == "high")
        
        dataset = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_samples': len(results),
                'complaints': complaints_count,
                'non_complaints': len(results) - complaints_count,
                'complaint_rate': complaints_count / len(results) if results else 0,
                'high_confidence_samples': high_conf_count,
                'high_confidence_rate': high_conf_count / len(results) if results else 0,
                'total_api_cost': self.total_cost,
                'average_cost_per_sample': self.total_cost / len(results) if results else 0,
                'labeling_method': 'gpt-4o-mini',
                'model_used': self.model,
                'description': 'inDriver reviews labeled by GPT-4o-mini for complaint classification'
            },
            'data': dataset_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Labeled dataset saved to {output_path}")
        return output_path

async def main():
    """Main labeling process"""
    print("🤖 GPT-4o-mini Professional Complaint Labeling for inDriver")
    print("=" * 60)
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='sk-...'")
        return 1
    
    try:
        labeler = ChatGPTLabeler()
        
        # Step 1: Get all reviews
        print("📥 Step 1: Fetching all inDriver reviews...")
        reviews = await labeler.get_all_reviews()
        
        if not reviews:
            print("❌ No reviews found in database")
            return 1
        
        print(f"✅ Found {len(reviews)} reviews to label")
        estimated_cost = len(reviews) * 0.0005  # Rough estimate: $0.0005 per review with GPT-4o-mini
        print(f"💰 Estimated cost: ~${estimated_cost:.2f} (GPT-4o-mini)")
        
        # Confirm before proceeding
        response = input(f"\n🤔 Label {len(reviews)} reviews using GPT-4o-mini? (y/N): ")
        if response.lower() != 'y':
            print("❌ Operation cancelled")
            return 0
        
        # Step 2: Label all reviews
        print(f"\n🏷️ Step 2: Labeling {len(reviews)} reviews with GPT-4o-mini...")
        print("⏱️ This will be much faster with GPT-4o-mini (10K requests/min)...")
        
        results = await labeler.label_batch_reviews(reviews, batch_size=50)
        
        # Step 3: Save results
        print("\n💾 Step 3: Saving labeled dataset...")
        output_path = labeler.save_labeled_dataset(results)
        
        # Summary
        complaints = sum(1 for r in results if r.is_complaint)
        high_conf = sum(1 for r in results if r.confidence == "high")
        
        print(f"\n🎯 Labeling Summary:")
        print(f"   📊 Total reviews: {len(results)}")
        print(f"   🚨 Complaints: {complaints} ({100*complaints/len(results):.1f}%)")
        print(f"   ✅ Non-complaints: {len(results) - complaints} ({100*(len(results)-complaints)/len(results):.1f}%)")
        print(f"   🎯 High confidence: {high_conf} ({100*high_conf/len(results):.1f}%)")
        print(f"   💰 Total cost: ${labeler.total_cost:.3f}")
        print(f"   📁 Saved to: {output_path}")
        
        print(f"\n🎉 Ready to train professional ML model!")
        return 0
        
    except Exception as e:
        logger.error(f"Labeling failed: {e}")
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))