#!/usr/bin/env python3
"""
Test alternative language detection methods for Kazakh language.

Compares different libraries and approaches for Kazakh language detection.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pavel.core.logger import get_logger

logger = get_logger(__name__)

def test_langdetect():
    """Test standard langdetect library"""
    try:
        from langdetect import detect, detect_langs
        logger.info("Testing langdetect library:")
        
        texts = [
            "Бұл қосымша өте жақсы, жылдам және ыңғайлы.",
            "Өте керемет қосымша! Жұмыс істейді тамаша.",
            "Жақсы ақша үнемдейтін қосымша. Ұсынамын!",
            "Мен бұл қосымшаны өте жақсы көремін.",
        ]
        
        for text in texts:
            try:
                result = detect(text)
                langs = detect_langs(text)
                logger.info(f"  '{text[:30]}...' → {result} {[f'{l.lang}:{l.prob:.2f}' for l in langs[:3]]}")
            except Exception as e:
                logger.info(f"  '{text[:30]}...' → ERROR: {e}")
                
        return True
    except ImportError:
        logger.warning("langdetect not available")
        return False

def test_polyglot():
    """Test polyglot library (if available)"""
    try:
        from polyglot.detect import Detector
        logger.info("\nTesting polyglot library:")
        
        texts = [
            "Бұл қосымша өте жақсы, жылдам және ыңғайлы.",
            "Өте керемет қосымша! Жұмыс істейді тамаша.",
        ]
        
        for text in texts:
            try:
                detector = Detector(text)
                logger.info(f"  '{text[:30]}...' → {detector.language.code} ({detector.language.confidence:.2f})")
            except Exception as e:
                logger.info(f"  '{text[:30]}...' → ERROR: {e}")
                
        return True
    except ImportError:
        logger.info("polyglot not available (pip install polyglot)")
        return False

def test_langid():
    """Test langid library"""
    try:
        import langid
        logger.info("\nTesting langid library:")
        
        texts = [
            "Бұл қосымша өте жақсы, жылдам және ыңғайлы.",
            "Өте керемет қосымша! Жұмыс істейді тамаша.",
            "Жақсы ақша үнемдейтін қосымша. Ұсынамын!",
        ]
        
        for text in texts:
            try:
                lang, conf = langid.classify(text)
                logger.info(f"  '{text[:30]}...' → {lang} ({conf:.2f})")
            except Exception as e:
                logger.info(f"  '{text[:30]}...' → ERROR: {e}")
                
        return True
    except ImportError:
        logger.info("langid not available (pip install langid)")
        return False

def test_textblob():
    """Test TextBlob library"""
    try:
        from textblob import TextBlob
        logger.info("\nTesting TextBlob library:")
        
        texts = [
            "Бұл қосымша өте жақсы, жылдам және ыңғайлы.",
            "Өте керемет қосымша! Жұмыс істейді тамаша.",
        ]
        
        for text in texts:
            try:
                blob = TextBlob(text)
                lang = blob.detect_language()
                logger.info(f"  '{text[:30]}...' → {lang}")
            except Exception as e:
                logger.info(f"  '{text[:30]}...' → ERROR: {e}")
                
        return True
    except ImportError:
        logger.info("TextBlob not available (pip install textblob)")
        return False

def test_googletrans():
    """Test Google Translate API detection"""
    try:
        from googletrans import Translator
        logger.info("\nTesting Google Translate detection:")
        
        translator = Translator()
        texts = [
            "Бұл қосымша өте жақсы, жылдам және ыңғайлы.",
            "Өте керемет қосымша! Жұмыс істейді тамаша.",
        ]
        
        for text in texts:
            try:
                result = translator.detect(text)
                logger.info(f"  '{text[:30]}...' → {result.lang} ({result.confidence:.2f})")
            except Exception as e:
                logger.info(f"  '{text[:30]}...' → ERROR: {e}")
                
        return True
    except ImportError:
        logger.info("googletrans not available (pip install googletrans==4.0.0-rc1)")
        return False

def test_fasttext():
    """Test FastText language identification"""
    try:
        import fasttext
        logger.info("\nTesting FastText library:")
        
        # Download model if needed
        model_path = "lid.176.bin"
        if not os.path.exists(model_path):
            logger.info("Downloading FastText language identification model...")
            import urllib.request
            urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", model_path)
        
        model = fasttext.load_model(model_path)
        
        texts = [
            "Бұл қосымша өте жақсы, жылдам және ыңғайлы.",
            "Өте керемет қосымша! Жұмыс істейді тамаша.",
            "Жақсы ақша үнемдейтін қосымша. Ұсынамын!",
        ]
        
        for text in texts:
            try:
                predictions = model.predict(text.replace('\n', ' '), k=3)
                langs = [lang.replace('__label__', '') for lang in predictions[0]]
                scores = predictions[1]
                result = list(zip(langs, scores))
                logger.info(f"  '{text[:30]}...' → {result}")
            except Exception as e:
                logger.info(f"  '{text[:30]}...' → ERROR: {e}")
                
        return True
    except ImportError:
        logger.info("fasttext not available (pip install fasttext)")
        return False

def test_spacy():
    """Test spaCy language detection"""
    try:
        import spacy
        from spacy.lang.kk import Kazakh  # Check if Kazakh is supported
        
        logger.info("\nTesting spaCy Kazakh support:")
        logger.info("  spaCy has Kazakh language class available!")
        
        # Check available models
        logger.info("  Available spaCy models:")
        import subprocess
        try:
            result = subprocess.run(['python', '-m', 'spacy', 'info'], capture_output=True, text=True)
            if 'kk' in result.stdout:
                logger.info("  Kazakh model found!")
            else:
                logger.info("  No Kazakh model installed")
        except:
            logger.info("  Could not check spaCy models")
            
        return True
    except ImportError:
        logger.info("spaCy not available (pip install spacy)")
        return False

def test_enhanced_patterns():
    """Test our enhanced pattern-based detection for Kazakh"""
    import re
    
    logger.info("\nTesting enhanced Kazakh patterns:")
    
    # More comprehensive Kazakh patterns
    kazakh_patterns = [
        # Kazakh-specific letters (must be present)
        re.compile(r'[әғқңөұүһі]', re.IGNORECASE),
        
        # Kazakh-specific letter combinations
        re.compile(r'(ғ[аәеиоөуүыі]|қ[аәеиоөуүыі]|ң[аәеиоөуүыі])', re.IGNORECASE),
        re.compile(r'(ә[йлмнрстт]|ө[йлмнрстт]|ұ[йлмнрстт]|ү[йлмнрстт])', re.IGNORECASE),
        
        # Common Kazakh words with unique letters
        re.compile(r'\b(бұл|және|үшін|жақсы|жаман|қосымша|өте|керемет|ұсынамын|қаржы|ақша)\b', re.IGNORECASE),
        
        # Kazakh morphology patterns (suffixes)
        re.compile(r'(ғы|ғі|ғу|ғү|қы|қі|қу|қү|ңы|ңі|ңу|ңү)', re.IGNORECASE),
        re.compile(r'(шы|ші|шу|шү|сы|сі|су|сү)', re.IGNORECASE),
        
        # Typical Kazakh sentence patterns
        re.compile(r'\b\w+[әөұү]\w*\s+\w+[ғқң]\w*\b', re.IGNORECASE),
    ]
    
    # Russian patterns for comparison
    russian_patterns = [
        # Russian-only patterns (no Kazakh letters)
        re.compile(r'\b(это|что|как|все|для|или|может|очень|хорошо|плохо|приложение|спасибо)\b', re.IGNORECASE),
        re.compile(r'[^әғқңөұүһі]*[аеиоуыэюя]{3,}[^әғқңөұүһі]*', re.IGNORECASE),  # Russian vowel patterns without Kazakh letters
    ]
    
    texts = [
        ("Бұл қосымша өте жақсы, жылдам және ыңғайлы.", "kk"),  # Kazakh
        ("Өте керемет қосымша! Жұмыс істейді тамаша.", "kk"),    # Kazakh
        ("Это приложение очень хорошее, быстрое и удобное.", "ru"),  # Russian
        ("Очень хорошо работает! Спасибо разработчикам.", "ru"),     # Russian
        ("Жақсы ақша үнемдейтін қосымша. Ұсынамын!", "kk"),      # Kazakh
    ]
    
    def detect_kazakh_enhanced(text):
        kazakh_score = 0
        russian_score = 0
        
        # Count Kazakh patterns
        for pattern in kazakh_patterns:
            matches = len(pattern.findall(text))
            kazakh_score += matches
        
        # Count Russian patterns  
        for pattern in russian_patterns:
            matches = len(pattern.findall(text))
            russian_score += matches
        
        # Normalize by text length
        text_len = len(text.split())
        kazakh_norm = kazakh_score / text_len if text_len > 0 else 0
        russian_norm = russian_score / text_len if text_len > 0 else 0
        
        if kazakh_norm > russian_norm and kazakh_score > 0:
            return 'kk', kazakh_norm
        elif russian_norm > kazakh_norm and russian_score > 0:
            return 'ru', russian_norm  
        else:
            return 'unknown', 0
    
    correct = 0
    total = len(texts)
    
    for text, expected in texts:
        detected, score = detect_kazakh_enhanced(text)
        is_correct = detected == expected
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        logger.info(f"  {status} '{text[:40]}...' → {detected} ({score:.2f}) [expected: {expected}]")
    
    accuracy = (correct / total * 100) if total > 0 else 0
    logger.info(f"\nEnhanced Kazakh patterns accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    return accuracy > 80

def main():
    """Test all available language detection libraries for Kazakh"""
    
    logger.info("🇰🇿 Testing Kazakh Language Detection Alternatives")
    logger.info("="*60)
    
    # Test all available libraries
    results = {}
    
    results['langdetect'] = test_langdetect()
    results['polyglot'] = test_polyglot() 
    results['langid'] = test_langid()
    results['textblob'] = test_textblob()
    results['googletrans'] = test_googletrans()
    results['fasttext'] = test_fasttext()
    results['spacy'] = test_spacy()
    results['enhanced_patterns'] = test_enhanced_patterns()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY OF ALTERNATIVES")
    logger.info("="*60)
    
    available_libs = [lib for lib, available in results.items() if available]
    logger.info(f"📚 Available libraries: {len(available_libs)}")
    for lib in available_libs:
        logger.info(f"  ✓ {lib}")
    
    unavailable_libs = [lib for lib, available in results.items() if not available]
    if unavailable_libs:
        logger.info(f"\n❌ Unavailable libraries: {len(unavailable_libs)}")
        for lib in unavailable_libs:
            logger.info(f"  ✗ {lib}")
    
    logger.info(f"\n🎯 Recommendations for Kazakh detection:")
    logger.info(f"  1. Enhanced pattern-based detection (custom implementation)")
    logger.info(f"  2. FastText (if available) - supports 176 languages including Kazakh")
    logger.info(f"  3. Google Translate API (requires internet, but very accurate)")
    logger.info(f"  4. Combine multiple methods for better accuracy")
    
    return len(available_libs) > 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)