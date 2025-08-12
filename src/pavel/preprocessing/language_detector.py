"""
Language detection for Google Play reviews.

Uses langdetect library with confidence scoring and fallback mechanisms.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

try:
    from langdetect import detect, detect_langs, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    LANGDETECT_AVAILABLE = True
    # Set seed for consistent results
    DetectorFactory.seed = 0
except ImportError:
    LANGDETECT_AVAILABLE = False

from pavel.core.logger import get_logger

logger = get_logger(__name__)

@dataclass
class LanguageDetectionResult:
    """Result of language detection"""
    language: str
    confidence: float
    method: str  # "langdetect", "patterns", "locale_fallback", "default"
    alternatives: List[Tuple[str, float]] = None  # Alternative languages with scores
    
    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []

class LanguageDetector:
    """
    Multi-method language detector for Google Play reviews.
    
    Features:
    - Primary: langdetect library with confidence scoring
    - Fallback: Pattern-based detection for common languages
    - Locale-based fallback using review metadata
    - Support for mixed-language content
    - Confidence thresholds for reliability
    """
    
    def __init__(self, 
                 min_confidence: float = 0.7,
                 min_text_length: int = 10,
                 use_locale_fallback: bool = True):
        self.min_confidence = min_confidence
        self.min_text_length = min_text_length
        self.use_locale_fallback = use_locale_fallback
        
        # Language mappings
        self._setup_language_mappings()
        
        # Pattern-based detection for common languages
        self._setup_language_patterns()
        
        if not LANGDETECT_AVAILABLE:
            logger.warning("langdetect library not available, using pattern-based detection only")
            
    def _setup_language_mappings(self):
        """Set up language code mappings and metadata"""
        
        # ISO 639-1 to full name mapping (expanded for inDriver markets)
        self.language_names = {
            'en': 'English',
            'ru': 'Russian', 
            'es': 'Spanish',
            'pt': 'Portuguese',
            'id': 'Indonesian',
            'kk': 'Kazakh',
            'ar': 'Arabic',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'tr': 'Turkish',
            'hi': 'Hindi',
            'ur': 'Urdu',
            'bn': 'Bengali',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'tl': 'Tagalog/Filipino',
            'ms': 'Malay',
            'uk': 'Ukrainian',
            'pl': 'Polish',
            'ro': 'Romanian',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'no': 'Norwegian',
            'da': 'Danish',
            'fi': 'Finnish',
            'he': 'Hebrew',
            'fa': 'Persian/Farsi',
            'az': 'Azerbaijani',
            'ky': 'Kyrgyz',
            'uz': 'Uzbek',
            'tg': 'Tajik',
            'ka': 'Georgian',
            'hy': 'Armenian',
            'ne': 'Nepali',
            'si': 'Sinhala',
            'ta': 'Tamil',
            'te': 'Telugu',
            'ml': 'Malayalam',
            'kn': 'Kannada',
            'gu': 'Gujarati',
            'mr': 'Marathi',
            'pa': 'Punjabi',
            'am': 'Amharic',
            'sw': 'Swahili',
            'yo': 'Yoruba',
            'ha': 'Hausa',
            'ig': 'Igbo',
            'zu': 'Zulu',
            'xh': 'Xhosa',
            'af': 'Afrikaans'
        }
        
        # Common locale to language mappings (expanded for inDriver markets)
        self.locale_to_language = {
            # English
            'en_US': 'en', 'en_GB': 'en', 'en_AU': 'en', 'en_CA': 'en', 'en_IN': 'en',
            'en_ZA': 'en', 'en_NG': 'en', 'en_KE': 'en', 'en_GH': 'en',
            # Russian
            'ru_RU': 'ru', 'ru_BY': 'ru', 'ru_KZ': 'ru', 'ru_UA': 'ru',
            # Spanish
            'es_ES': 'es', 'es_MX': 'es', 'es_AR': 'es', 'es_CO': 'es', 'es_CL': 'es',
            'es_PE': 'es', 'es_VE': 'es', 'es_EC': 'es', 'es_BO': 'es', 'es_UY': 'es',
            'es_PY': 'es', 'es_CR': 'es', 'es_PA': 'es', 'es_DO': 'es', 'es_GT': 'es',
            'es_HN': 'es', 'es_SV': 'es', 'es_NI': 'es',
            # Portuguese
            'pt_BR': 'pt', 'pt_PT': 'pt', 'pt_AO': 'pt', 'pt_MZ': 'pt',
            # Arabic
            'ar_SA': 'ar', 'ar_EG': 'ar', 'ar_AE': 'ar', 'ar_KW': 'ar', 'ar_QA': 'ar',
            'ar_BH': 'ar', 'ar_OM': 'ar', 'ar_JO': 'ar', 'ar_LB': 'ar', 'ar_IQ': 'ar',
            'ar_SY': 'ar', 'ar_YE': 'ar', 'ar_LY': 'ar', 'ar_TN': 'ar', 'ar_DZ': 'ar',
            'ar_MA': 'ar', 'ar_SD': 'ar',
            # Central Asian languages
            'kk_KZ': 'kk', 'ky_KG': 'ky', 'uz_UZ': 'uz', 'tg_TJ': 'tg',
            'az_AZ': 'az', 'tk_TM': 'tk',
            # South Asian languages
            'hi_IN': 'hi', 'ur_PK': 'ur', 'ur_IN': 'ur', 'bn_BD': 'bn', 'bn_IN': 'bn',
            'pa_IN': 'pa', 'pa_PK': 'pa', 'gu_IN': 'gu', 'mr_IN': 'mr', 'ta_IN': 'ta',
            'te_IN': 'te', 'kn_IN': 'kn', 'ml_IN': 'ml', 'ne_NP': 'ne', 'si_LK': 'si',
            # Southeast Asian languages
            'id_ID': 'id', 'ms_MY': 'ms', 'th_TH': 'th', 'vi_VN': 'vi', 'tl_PH': 'tl',
            'my_MM': 'my', 'km_KH': 'km', 'lo_LA': 'lo',
            # East Asian languages
            'zh_CN': 'zh', 'zh_TW': 'zh', 'zh_HK': 'zh', 'ja_JP': 'ja', 'ko_KR': 'ko',
            # European languages
            'fr_FR': 'fr', 'fr_CA': 'fr', 'fr_BE': 'fr', 'fr_CH': 'fr',
            'de_DE': 'de', 'de_AT': 'de', 'de_CH': 'de',
            'it_IT': 'it', 'it_CH': 'it',
            'tr_TR': 'tr', 'pl_PL': 'pl', 'uk_UA': 'uk', 'ro_RO': 'ro',
            'nl_NL': 'nl', 'nl_BE': 'nl',
            'sv_SE': 'sv', 'no_NO': 'no', 'da_DK': 'da', 'fi_FI': 'fi',
            # Other languages
            'he_IL': 'he', 'fa_IR': 'fa', 'ka_GE': 'ka', 'hy_AM': 'hy',
            'am_ET': 'am', 'sw_KE': 'sw', 'sw_TZ': 'sw',
            'yo_NG': 'yo', 'ha_NG': 'ha', 'ig_NG': 'ig',
            'zu_ZA': 'zu', 'xh_ZA': 'xh', 'af_ZA': 'af'
        }
        
    def _setup_language_patterns(self):
        """Set up regex patterns for basic language detection"""
        
        self.language_patterns = {
            # Cyrillic-based languages
            'ru': [
                # Russian Cyrillic (no special Kazakh letters)
                re.compile(r'[а-яё]+', re.IGNORECASE),
                # Common Russian words
                re.compile(r'\b(это|что|как|все|для|или|может|очень|хорошо|плохо|приложение|спасибо|пожалуйста)\b', re.IGNORECASE)
            ],
            'kk': [
                # Kazakh-specific Cyrillic letters
                re.compile(r'[әғқңөұүһі]+', re.IGNORECASE),
                # Common Kazakh words
                re.compile(r'\b(бұл|және|үшін|жақсы|жаман|қосымша|өте|керемет)\b', re.IGNORECASE)
            ],
            'uk': [
                # Ukrainian-specific letters
                re.compile(r'[ґєіїє]+', re.IGNORECASE),
                # Common Ukrainian words
                re.compile(r'\b(це|що|як|для|дуже|добре|погано|додаток)\b', re.IGNORECASE)
            ],
            
            # Arabic script languages
            'ar': [
                # Arabic script
                re.compile(r'[\u0600-\u06FF]+'),  # Arabic Unicode range
                # Common Arabic words
                re.compile(r'\b(هذا|جيد|سيء|تطبيق|شكرا|ممتاز|رائع)\b')
            ],
            'fa': [
                # Persian/Farsi specific characters
                re.compile(r'[پچژگ]+'),
                # Common Persian words
                re.compile(r'\b(این|خوب|بد|برنامه|متشکرم|عالی)\b')
            ],
            'ur': [
                # Urdu script (Arabic + special characters)
                re.compile(r'[ٹڈڑںھےۓ]+'),
                # Common Urdu words
                re.compile(r'\b(یہ|اچھا|برا|ایپ|شکریہ)\b')
            ],
            
            # Latin script languages
            'en': [
                # Common English words
                re.compile(r'\b(the|and|that|have|for|not|with|you|this|but|app|good|bad|great|excellent)\b', re.IGNORECASE)
            ],
            'es': [
                # Spanish patterns
                re.compile(r'\b(que|con|por|para|una|como|más|pero|muy|bien|mal|aplicación|gracias|excelente)\b', re.IGNORECASE)
            ],
            'pt': [
                # Portuguese patterns  
                re.compile(r'\b(que|com|por|para|uma|como|mais|mas|muito|bem|mal|aplicativo|obrigado|ótimo)\b', re.IGNORECASE)
            ],
            'id': [
                # Indonesian patterns
                re.compile(r'\b(yang|dan|untuk|ini|dari|dengan|tidak|adalah|atau|sangat|baik|buruk|aplikasi|terima kasih|bagus)\b', re.IGNORECASE)
            ],
            'tr': [
                # Turkish patterns with special characters
                re.compile(r'[şğıüöç]+', re.IGNORECASE),
                # Common Turkish words
                re.compile(r'\b(bu|ve|için|çok|iyi|kötü|uygulama|teşekkür|mükemmel)\b', re.IGNORECASE)
            ],
            
            # Asian languages with unique scripts
            'th': [
                # Thai script
                re.compile(r'[\u0E00-\u0E7F]+'),  # Thai Unicode range
                # Common Thai words
                re.compile(r'\b(ดี|ไม่ดี|แอพ|ขอบคุณ)\b')
            ],
            'hi': [
                # Devanagari script (Hindi)
                re.compile(r'[\u0900-\u097F]+'),  # Devanagari Unicode range
                # Common Hindi words
                re.compile(r'\b(यह|अच्छा|बुरा|ऐप|धन्यवाद|बहुत)\b')
            ],
            'bn': [
                # Bengali script
                re.compile(r'[\u0980-\u09FF]+'),  # Bengali Unicode range
                # Common Bengali words
                re.compile(r'\b(এই|ভাল|খারাপ|অ্যাপ|ধন্যবাদ)\b')
            ],
            'vi': [
                # Vietnamese with tone marks
                re.compile(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+', re.IGNORECASE),
                # Common Vietnamese words
                re.compile(r'\b(này|tốt|xấu|ứng dụng|cảm ơn|rất)\b', re.IGNORECASE)
            ],
            'zh': [
                # Chinese characters
                re.compile(r'[\u4E00-\u9FFF]+'),  # CJK Unified Ideographs
                # Common Chinese words
                re.compile(r'(好|不好|应用|谢谢|很)')
            ],
            'ja': [
                # Japanese scripts (Hiragana, Katakana)
                re.compile(r'[\u3040-\u309F\u30A0-\u30FF]+'),
                # Common Japanese words
                re.compile(r'(いい|よくない|アプリ|ありがとう)')
            ],
            'ko': [
                # Korean Hangul
                re.compile(r'[\uAC00-\uD7AF]+'),  # Hangul syllables
                # Common Korean words
                re.compile(r'(좋아|나빠|앱|감사합니다)')
            ]
        }
        
    def detect_by_patterns(self, text: str) -> Optional[LanguageDetectionResult]:
        """
        Detect language using regex patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            LanguageDetectionResult or None if no pattern matches
        """
        if not text or len(text) < 3:
            return None
            
        text = text.lower()
        pattern_scores = {}
        
        for lang_code, patterns in self.language_patterns.items():
            score = 0
            total_matches = 0
            
            for pattern in patterns:
                matches = len(pattern.findall(text))
                score += matches
                total_matches += matches
                
            if total_matches > 0:
                # Normalize score by text length
                pattern_scores[lang_code] = score / len(text.split())
                
        if not pattern_scores:
            return None
            
        # Get best match
        best_lang = max(pattern_scores, key=pattern_scores.get)
        confidence = min(pattern_scores[best_lang] * 2, 1.0)  # Scale to 0-1
        
        return LanguageDetectionResult(
            language=best_lang,
            confidence=confidence,
            method="patterns",
            alternatives=[(lang, score) for lang, score in pattern_scores.items() if lang != best_lang]
        )
        
    def detect_by_langdetect(self, text: str) -> Optional[LanguageDetectionResult]:
        """
        Detect language using langdetect library.
        
        Args:
            text: Text to analyze
            
        Returns:
            LanguageDetectionResult or None if detection fails
        """
        if not LANGDETECT_AVAILABLE or not text or len(text) < self.min_text_length:
            return None
            
        try:
            # Get multiple language predictions with confidence
            lang_probs = detect_langs(text)
            
            if not lang_probs:
                return None
                
            # Get the best prediction
            best = lang_probs[0]
            
            # Create alternatives list
            alternatives = [(lang.lang, lang.prob) for lang in lang_probs[1:5]]  # Top 5 alternatives
            
            return LanguageDetectionResult(
                language=best.lang,
                confidence=best.prob,
                method="langdetect",
                alternatives=alternatives
            )
            
        except LangDetectException as e:
            logger.debug(f"langdetect failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected langdetect error: {e}")
            return None
            
    def detect_by_locale(self, locale: Optional[str]) -> Optional[LanguageDetectionResult]:
        """
        Detect language based on locale information.
        
        Args:
            locale: Locale string (e.g., "en_US", "ru_RU")
            
        Returns:
            LanguageDetectionResult or None if locale not recognized
        """
        if not locale or not self.use_locale_fallback:
            return None
            
        language = self.locale_to_language.get(locale.replace('-', '_'))
        
        if language:
            return LanguageDetectionResult(
                language=language,
                confidence=0.6,  # Medium confidence for locale-based detection
                method="locale_fallback"
            )
            
        return None
        
    def detect(self, text: str, locale: Optional[str] = None) -> LanguageDetectionResult:
        """
        Detect language using multiple methods with fallback.
        
        Args:
            text: Text to analyze
            locale: Optional locale information for fallback
            
        Returns:
            LanguageDetectionResult with best detection
        """
        # Try langdetect first (most accurate)
        result = self.detect_by_langdetect(text)
        
        if result and result.confidence >= self.min_confidence:
            logger.debug(f"Language detected by langdetect: {result.language} ({result.confidence:.2f})")
            return result
            
        # Try pattern-based detection
        pattern_result = self.detect_by_patterns(text)
        
        if pattern_result and pattern_result.confidence >= 0.3:  # Lower threshold for patterns
            logger.debug(f"Language detected by patterns: {pattern_result.language} ({pattern_result.confidence:.2f})")
            return pattern_result
            
        # Try locale fallback
        locale_result = self.detect_by_locale(locale)
        
        if locale_result:
            logger.debug(f"Language detected by locale: {locale_result.language}")
            return locale_result
            
        # Default to English if all methods fail
        logger.debug("Language detection failed, defaulting to English")
        return LanguageDetectionResult(
            language="en",
            confidence=0.1,
            method="default"
        )
        
    def batch_detect(self, texts: List[str], locales: Optional[List[str]] = None) -> List[LanguageDetectionResult]:
        """
        Detect languages for multiple texts.
        
        Args:
            texts: List of texts to analyze
            locales: Optional list of locales (same length as texts)
            
        Returns:
            List of LanguageDetectionResults
        """
        results = []
        
        if locales is None:
            locales = [None] * len(texts)
        elif len(locales) != len(texts):
            logger.warning("Locales list length doesn't match texts list, ignoring locales")
            locales = [None] * len(texts)
            
        for text, locale in zip(texts, locales):
            result = self.detect(text, locale)
            results.append(result)
            
        return results
        
    def get_language_stats(self, results: List[LanguageDetectionResult]) -> Dict:
        """
        Get statistics from language detection results.
        
        Args:
            results: List of LanguageDetectionResults
            
        Returns:
            Dictionary with language statistics
        """
        languages = [r.language for r in results]
        methods = [r.method for r in results]
        confidences = [r.confidence for r in results]
        
        return {
            "total_texts": len(results),
            "language_distribution": dict(Counter(languages)),
            "method_distribution": dict(Counter(methods)),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "low_confidence_count": len([c for c in confidences if c < self.min_confidence]),
            "supported_languages": list(self.language_names.keys())
        }
        
    def is_supported_language(self, language: str) -> bool:
        """Check if language is in supported list"""
        return language in self.language_names
        
    def get_language_name(self, language_code: str) -> str:
        """Get full language name from code"""
        return self.language_names.get(language_code, language_code)