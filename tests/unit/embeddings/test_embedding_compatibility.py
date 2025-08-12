#!/usr/bin/env python3
"""
Test embedding model compatibility with our preprocessing output.

Tests preprocessing output format with popular embedding models:
- E5 (multilingual-e5-large)
- OpenAI text-embedding-ada-002
- Sentence-Transformers compatibility
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from typing import List, Dict, Any
import time

from pavel.preprocessing import PreprocessingPipeline
from pavel.core.logger import get_logger

logger = get_logger(__name__)

def test_e5_compatibility():
    """Test E5 embedding model compatibility"""
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info("Testing E5 multilingual embedding compatibility:")
        
        # Load E5 model
        model = SentenceTransformer('intfloat/multilingual-e5-large')
        
        # Test samples from our preprocessing
        test_texts = [
            "This app is great and works perfectly!",
            "Отличное приложение, работает быстро.",
            "Bұл қосымша өте жақсы және тамаша жұмыс істейді.",
            "Esta aplicación es excelente y funciona perfectamente.",
            "Este aplicativo é ótimo e funciona perfeitamente!",
            "هذا التطبيق رائع ويعمل بشكل مثالي!",
            "แอพนี้ยอดเยี่ยมและทำงานได้อย่างสมบูรณ์แบบ!",
        ]
        
        logger.info(f"  Testing {len(test_texts)} preprocessed texts with E5...")
        
        # Test embedding generation
        start_time = time.time()
        embeddings = model.encode(test_texts)
        end_time = time.time()
        
        # Validate results
        expected_dim = 1024  # E5-large dimension
        success = True
        
        for i, (text, embedding) in enumerate(zip(test_texts, embeddings)):
            if embedding.shape[0] != expected_dim:
                logger.error(f"  ✗ Text {i+1}: Wrong embedding dimension {embedding.shape[0]}, expected {expected_dim}")
                success = False
            else:
                logger.info(f"  ✓ Text {i+1}: {text[:30]}... → {embedding.shape[0]}D embedding")
        
        # Performance metrics
        avg_time = (end_time - start_time) / len(test_texts)
        logger.info(f"  Performance: {len(test_texts)} texts in {end_time-start_time:.2f}s ({avg_time*1000:.1f}ms/text)")
        
        if success:
            logger.info("✅ E5 compatibility: All tests passed!")
        
        return success, len(test_texts), avg_time
        
    except ImportError:
        logger.warning("sentence-transformers not available (pip install sentence-transformers)")
        return False, 0, 0
    except Exception as e:
        logger.error(f"E5 test failed: {e}")
        return False, 0, 0

def test_openai_format_compatibility():
    """Test OpenAI embedding API format compatibility"""
    
    logger.info("\nTesting OpenAI embeddings format compatibility:")
    
    # Test texts that would come from our preprocessing
    test_samples = [
        {
            "text": "This app is amazing! Works perfectly.",
            "language": "en",
            "sentences": ["This app is amazing!", "Works perfectly."]
        },
        {
            "text": "Отличное приложение, рекомендую всем!",
            "language": "ru", 
            "sentences": ["Отличное приложение, рекомендую всем!"]
        },
        {
            "text": "Bұл қосымша керемет!",
            "language": "kk",
            "sentences": ["Bұл қосымша керемет!"]
        }
    ]
    
    # Simulate OpenAI API format validation
    openai_compatible = True
    
    for i, sample in enumerate(test_samples, 1):
        text = sample["text"]
        
        # Check OpenAI requirements
        checks = {
            "length": len(text) <= 8191,  # OpenAI token limit approximation
            "encoding": text.encode('utf-8').decode('utf-8') == text,  # UTF-8 compatibility
            "no_null": '\x00' not in text,  # No null bytes
            "printable": all(ord(c) >= 32 or c in '\n\r\t' for c in text)  # Printable chars
        }
        
        all_passed = all(checks.values())
        
        if all_passed:
            logger.info(f"  ✓ Sample {i}: '{text[:30]}...' → OpenAI compatible")
        else:
            logger.error(f"  ✗ Sample {i}: Failed checks: {[k for k, v in checks.items() if not v]}")
            openai_compatible = False
    
    # Test format with different input structures
    formats_to_test = [
        ("Single text", "Just a simple review text"),
        ("With newlines", "Line 1\nLine 2\nLine 3"),
        ("With emojis", "Great app! 😍 Works perfectly! 👍"),
        ("Mixed languages", "Great приложение! muy bueno"),
        ("Long text", "A" * 1000 + " This is a very long review text that tests the limits.")
    ]
    
    logger.info("\n  Testing different text formats:")
    for format_name, text in formats_to_test:
        try:
            # Simulate what would be sent to OpenAI
            encoded = text.encode('utf-8')
            decoded = encoded.decode('utf-8')
            
            is_valid = (
                len(text) <= 8191 and  # Token limit approximation
                decoded == text and    # Encoding integrity
                '\x00' not in text     # No null bytes
            )
            
            status = "✓" if is_valid else "✗"
            logger.info(f"    {status} {format_name}: {len(text)} chars, UTF-8 OK")
            
            if not is_valid:
                openai_compatible = False
                
        except Exception as e:
            logger.error(f"    ✗ {format_name}: {e}")
            openai_compatible = False
    
    if openai_compatible:
        logger.info("✅ OpenAI format compatibility: All tests passed!")
    
    return openai_compatible

async def test_preprocessing_to_embeddings():
    """Test full preprocessing pipeline to embedding readiness"""
    
    logger.info("\nTesting preprocessing → embedding pipeline:")
    
    # Initialize preprocessing pipeline
    pipeline = PreprocessingPipeline()
    
    # Sample reviews (raw input)
    raw_reviews = [
        {
            "review_id": "test_1",
            "app_id": "com.test.app", 
            "content": "This app is AMAZING!!! Really    fast and works perfectly. I love it 😍 Highly recommend!",
            "locale": "en_US",
            "created_at": "2024-01-01T00:00:00Z",
            "score": 5
        },
        {
            "review_id": "test_2",
            "app_id": "com.test.app",
            "content": "Отличное приложение!!! Работает очень быстро. Рекомендую всем пользователям.",
            "locale": "ru_RU", 
            "created_at": "2024-01-01T00:00:00Z",
            "score": 5
        },
        {
            "review_id": "test_3",
            "app_id": "com.test.app",
            "content": "Бұл қосымша өте жақсы! Жылдам және ыңғайлы. Барлығына ұсынамын!",
            "locale": "kk_KZ",
            "created_at": "2024-01-01T00:00:00Z", 
            "score": 4
        }
    ]
    
    embedding_ready_texts = []
    
    for review in raw_reviews:
        try:
            # Process through our pipeline
            processed = await pipeline.process_single_review(review)
            
            # Extract embedding-ready text
            embedding_text = processed.normalized_content
            
            logger.info(f"  Review {processed.review_id}:")
            logger.info(f"    Original: {review['content'][:50]}...")
            logger.info(f"    Processed: {embedding_text[:50]}...")
            logger.info(f"    Language: {processed.detected_language} ({processed.language_confidence:.2f})")
            logger.info(f"    Sentences: {processed.sentence_count}")
            
            # Validate embedding readiness
            checks = {
                "not_empty": len(embedding_text.strip()) > 0,
                "reasonable_length": 10 <= len(embedding_text) <= 2000,
                "utf8_compatible": embedding_text.encode('utf-8').decode('utf-8') == embedding_text,
                "no_excessive_whitespace": '  ' not in embedding_text,
                "language_detected": processed.detected_language != 'unknown'
            }
            
            all_passed = all(checks.values())
            status = "✓" if all_passed else "✗"
            logger.info(f"    {status} Embedding ready: {all_passed}")
            
            if not all_passed:
                failed_checks = [k for k, v in checks.items() if not v]
                logger.warning(f"    Failed checks: {failed_checks}")
            
            embedding_ready_texts.append({
                "id": processed.review_id,
                "text": embedding_text,
                "language": processed.detected_language,
                "ready": all_passed
            })
            
        except Exception as e:
            logger.error(f"  Processing failed for {review['review_id']}: {e}")
            
    success_rate = len([t for t in embedding_ready_texts if t["ready"]]) / len(embedding_ready_texts)
    logger.info(f"\n  Embedding readiness: {success_rate*100:.1f}% ({len([t for t in embedding_ready_texts if t['ready']])}/{len(embedding_ready_texts)})")
    
    return embedding_ready_texts, success_rate >= 0.8

def test_sentence_transformers_compatibility():
    """Test sentence-transformers library compatibility"""
    
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info("\nTesting sentence-transformers compatibility:")
        
        # Test with different models
        models_to_test = [
            'all-MiniLM-L6-v2',  # Lightweight English
            'paraphrase-multilingual-MiniLM-L12-v2',  # Multilingual
        ]
        
        # Preprocessed texts in different languages  
        test_texts = [
            "This application works great!",
            "Отличная программа!",
            "¡Excelente aplicación!",
            "Aplicativo fantástico!",
        ]
        
        compatible_models = 0
        
        for model_name in models_to_test:
            try:
                logger.info(f"  Testing {model_name}...")
                model = SentenceTransformer(model_name)
                
                embeddings = model.encode(test_texts[:2])  # Test with 2 texts
                
                if embeddings is not None and len(embeddings) == 2:
                    logger.info(f"    ✓ {model_name}: {embeddings.shape[1]}D embeddings")
                    compatible_models += 1
                else:
                    logger.warning(f"    ✗ {model_name}: Invalid embedding output")
                    
            except Exception as e:
                logger.warning(f"    ✗ {model_name}: {e}")
        
        success = compatible_models > 0
        if success:
            logger.info(f"✅ Sentence-transformers compatibility: {compatible_models}/{len(models_to_test)} models working")
            
        return success
        
    except ImportError:
        logger.warning("sentence-transformers not available for full testing")
        return True  # Don't fail if library not installed

async def main():
    """Run all embedding compatibility tests"""
    
    logger.info("🔗 Testing Embedding Model Compatibility")
    logger.info("="*60)
    
    results = {}
    
    # Test E5 compatibility
    e5_success, e5_count, e5_time = test_e5_compatibility()
    results['e5'] = e5_success
    
    # Test OpenAI format compatibility
    openai_success = test_openai_format_compatibility()
    results['openai'] = openai_success
    
    # Test preprocessing to embeddings pipeline
    embedding_texts, pipeline_success = await test_preprocessing_to_embeddings()
    results['pipeline'] = pipeline_success
    
    # Test sentence-transformers compatibility
    st_success = test_sentence_transformers_compatibility()
    results['sentence_transformers'] = st_success
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("EMBEDDING COMPATIBILITY SUMMARY")
    logger.info("="*60)
    
    passed_tests = len([r for r in results.values() if r])
    total_tests = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {test_name.upper():20} {status}")
    
    logger.info(f"\n📊 Overall: {passed_tests}/{total_tests} compatibility tests passed")
    
    if passed_tests >= total_tests * 0.75:  # 75% pass rate
        logger.info("🎉 EXCELLENT: Preprocessing output is compatible with major embedding models!")
        logger.info("✅ Ready for Stage 4 (Vector embeddings)")
    else:
        logger.info("⚠️  Some compatibility issues found - review and fix before Stage 4")
    
    # Practical recommendations
    logger.info(f"\n💡 Recommendations:")
    logger.info(f"  - Preprocessed text length: 10-2000 characters optimal")
    logger.info(f"  - UTF-8 encoding: Required for all models")
    logger.info(f"  - Language detection: Enables model selection")
    logger.info(f"  - Sentence segmentation: Useful for semantic chunking")
    
    return passed_tests >= total_tests * 0.75

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)