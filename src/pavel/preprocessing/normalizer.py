"""
Text normalization for Google Play reviews.

Handles Unicode normalization, HTML entities, emojis, and text cleanup.
"""

import re
import unicodedata
import html
from typing import Dict, List, Optional
from dataclasses import dataclass

from pavel.core.logger import get_logger

logger = get_logger(__name__)

@dataclass
class NormalizationStats:
    """Statistics from text normalization"""
    original_length: int = 0
    normalized_length: int = 0
    html_entities_found: int = 0
    emojis_found: int = 0
    urls_removed: int = 0
    repeated_chars_normalized: int = 0
    unicode_normalized: bool = False

class TextNormalizer:
    """
    Comprehensive text normalizer for Google Play reviews.
    
    Features:
    - Unicode normalization (NFC)
    - HTML entity decoding
    - URL removal
    - Excessive whitespace cleanup
    - Repeated character normalization
    - Emoji handling (preserve or remove)
    - Special character cleanup
    """
    
    def __init__(self, 
                 preserve_emojis: bool = True,
                 max_repeated_chars: int = 3,
                 remove_urls: bool = True):
        self.preserve_emojis = preserve_emojis
        self.max_repeated_chars = max_repeated_chars
        self.remove_urls = remove_urls
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile frequently used regex patterns"""
        # URL patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Repeated characters (more than max_repeated_chars)
        self.repeated_char_pattern = re.compile(r'(.)\1{' + str(self.max_repeated_chars) + ',}')
        
        # Multiple whitespace
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Emoji pattern (basic Unicode ranges)
        self.emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F]|'  # emoticons
            r'[\U0001F300-\U0001F5FF]|'  # symbols & pictographs
            r'[\U0001F680-\U0001F6FF]|'  # transport & map
            r'[\U0001F1E0-\U0001F1FF]|'  # flags
            r'[\U00002700-\U000027BF]|'  # dingbats
            r'[\U0001F900-\U0001F9FF]|'  # supplemental symbols
            r'[\U00002600-\U000026FF]'   # miscellaneous symbols
        )
        
        # HTML tags
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        
        # Excessive punctuation
        self.punct_pattern = re.compile(r'[.!?]{4,}')
        
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode to NFC form"""
        if not text:
            return text
            
        # Normalize to NFC (Canonical Decomposition + Canonical Composition)
        return unicodedata.normalize('NFC', text)
        
    def decode_html_entities(self, text: str) -> str:
        """Decode HTML entities like &amp;, &lt;, etc."""
        if not text:
            return text
            
        return html.unescape(text)
        
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags"""
        if not text:
            return text
            
        return self.html_tag_pattern.sub('', text)
        
    def normalize_whitespace(self, text: str) -> str:
        """Normalize excessive whitespace"""
        if not text:
            return text
            
        # Replace multiple whitespace with single space
        text = self.whitespace_pattern.sub(' ', text)
        
        # Strip leading/trailing whitespace
        return text.strip()
        
    def normalize_repeated_chars(self, text: str) -> str:
        """Normalize excessive repeated characters"""
        if not text:
            return text
            
        def replace_repeated(match):
            char = match.group(1)
            return char * self.max_repeated_chars
            
        return self.repeated_char_pattern.sub(replace_repeated, text)
        
    def remove_urls_from_text(self, text: str) -> str:
        """Remove URLs from text"""
        if not text:
            return text
            
        return self.url_pattern.sub('', text)
        
    def handle_emojis(self, text: str) -> str:
        """Handle emojis (preserve or remove)"""
        if not text:
            return text
            
        if self.preserve_emojis:
            return text
        else:
            return self.emoji_pattern.sub('', text)
            
    def normalize_punctuation(self, text: str) -> str:
        """Normalize excessive punctuation"""
        if not text:
            return text
            
        # Replace excessive punctuation with max 3 characters
        text = self.punct_pattern.sub(lambda m: m.group(0)[:3], text)
        
        return text
        
    def normalize(self, text: str) -> tuple[str, NormalizationStats]:
        """
        Apply full normalization pipeline to text.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Tuple of (normalized_text, stats)
        """
        if not text or not isinstance(text, str):
            return "", NormalizationStats()
            
        original_text = text
        stats = NormalizationStats(original_length=len(text))
        
        # Count elements before normalization
        stats.html_entities_found = len(re.findall(r'&[a-zA-Z0-9#]+;', text))
        stats.emojis_found = len(self.emoji_pattern.findall(text))
        if self.remove_urls:
            stats.urls_removed = len(self.url_pattern.findall(text))
        
        # Apply normalization steps
        logger.debug(f"Normalizing text: {text[:100]}...")
        
        # 1. Unicode normalization
        text = self.normalize_unicode(text)
        stats.unicode_normalized = True
        
        # 2. Remove HTML tags FIRST (before entity decoding to avoid conflicts)
        text = self.remove_html_tags(text)
        
        # 3. Decode HTML entities (after tag removal)
        text = self.decode_html_entities(text)
        
        # 4. Remove URLs (optional)
        if self.remove_urls:
            text = self.remove_urls_from_text(text)
            
        # 5. Handle emojis
        text = self.handle_emojis(text)
        
        # 6. Normalize repeated characters
        before_repeated = text
        text = self.normalize_repeated_chars(text)
        stats.repeated_chars_normalized = len(before_repeated) - len(text)
        
        # 7. Normalize punctuation
        text = self.normalize_punctuation(text)
        
        # 8. Normalize whitespace (final step)
        text = self.normalize_whitespace(text)
        
        stats.normalized_length = len(text)
        
        logger.debug(f"Normalized to: {text}")
        
        return text, stats
        
    def batch_normalize(self, texts: List[str]) -> List[tuple[str, NormalizationStats]]:
        """
        Normalize multiple texts in batch.
        
        Args:
            texts: List of texts to normalize
            
        Returns:
            List of (normalized_text, stats) tuples
        """
        results = []
        
        for text in texts:
            normalized, stats = self.normalize(text)
            results.append((normalized, stats))
            
        return results
        
    def get_settings(self) -> Dict:
        """Get current normalizer settings"""
        return {
            "preserve_emojis": self.preserve_emojis,
            "max_repeated_chars": self.max_repeated_chars,
            "remove_urls": self.remove_urls
        }
        
    def is_meaningful_text(self, text: str, min_length: int = 3) -> bool:
        """
        Check if text is meaningful after normalization.
        
        Args:
            text: Text to check
            min_length: Minimum meaningful length
            
        Returns:
            True if text is meaningful
        """
        if not text:
            return False
            
        normalized, _ = self.normalize(text)
        
        # Check length
        if len(normalized.strip()) < min_length:
            return False
            
        # Check if it's not just punctuation or whitespace
        if re.match(r'^[\s\W]*$', normalized):
            return False
            
        return True