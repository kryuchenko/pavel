"""
Sentence splitting for Google Play reviews.

Handles multilingual sentence segmentation with special handling for:
- App reviews (short, informal text)
- Multiple languages
- Emojis and special characters
- Abbreviations and edge cases
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from pavel.core.logger import get_logger

logger = get_logger(__name__)

@dataclass
class SentenceSplitResult:
    """Result of sentence splitting"""
    sentences: List[str]
    original_text: str
    method: str  # "regex", "spacy", "simple"
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    
    def __post_init__(self):
        self.sentence_count = len(self.sentences)
        if self.sentences:
            self.avg_sentence_length = sum(len(s) for s in self.sentences) / len(self.sentences)

class SentenceSplitter:
    """
    Multilingual sentence splitter optimized for app reviews.
    
    Features:
    - Language-aware splitting
    - Emoji and special character handling
    - Abbreviation protection
    - Review-specific patterns (ratings, app names)
    - Minimum sentence length filtering
    """
    
    def __init__(self, 
                 min_sentence_length: int = 5,
                 max_sentence_length: int = 500,
                 preserve_emojis: bool = True):
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.preserve_emojis = preserve_emojis
        
        self._setup_patterns()
        self._setup_abbreviations()
        
    def _setup_patterns(self):
        """Set up regex patterns for sentence splitting"""
        
        # Basic sentence terminators
        self.sentence_terminators = r'[.!?]+'
        
        # Emoji patterns (to handle as sentence boundaries in some cases)
        self.emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F]|'  # emoticons
            r'[\U0001F300-\U0001F5FF]|'  # symbols & pictographs
            r'[\U0001F680-\U0001F6FF]|'  # transport & map
            r'[\U0001F1E0-\U0001F1FF]|'  # flags
            r'[\U00002700-\U000027BF]|'  # dingbats
            r'[\U0001F900-\U0001F9FF]|'  # supplemental symbols
            r'[\U00002600-\U000026FF]'   # miscellaneous symbols
        )
        
        # Sentence boundary patterns with lookbehind/lookahead
        self.sentence_pattern = re.compile(
            r'(?<=[.!?])\s+',  # Split after punctuation followed by whitespace
            re.MULTILINE
        )
        
        # Line break patterns (for reviews with explicit line breaks)
        self.line_break_pattern = re.compile(r'\n\s*\n')  # Double line breaks
        
        # Review-specific patterns
        self.rating_pattern = re.compile(r'\b[1-5]\s*stars?\b', re.IGNORECASE)
        self.app_mention_pattern = re.compile(r'\b(app|application|program)\b', re.IGNORECASE)
        
    def _setup_abbreviations(self):
        """Set up common abbreviations that shouldn't split sentences"""
        
        # Common abbreviations by language
        self.abbreviations = {
            'en': {
                'dr', 'mr', 'mrs', 'ms', 'prof', 'inc', 'ltd', 'corp', 'co',
                'etc', 'vs', 'e.g', 'i.e', 'p.s', 'a.m', 'p.m', 'u.s', 'u.k',
                'min', 'max', 'sec', 'hrs'
            },
            'ru': {
                'др', 'г', 'гр', 'проф', 'и.т.д', 'т.е', 'т.к', 'т.п',
                'руб', 'коп', 'тыс', 'млн', 'млрд', 'см', 'км', 'кг'
            },
            'es': {
                'dr', 'sra', 'sr', 'etc', 'pág', 'vs', 'ej', 'p.ej',
                'min', 'máx', 'seg', 'hrs'
            },
            'pt': {
                'dr', 'sra', 'sr', 'etc', 'pág', 'vs', 'ex', 'p.ex',
                'min', 'máx', 'seg', 'hrs'
            },
            'id': {
                'dr', 'drs', 'prof', 'dll', 'dsb', 'pt', 'cv', 'tbk',
                'mnt', 'dtk', 'jam'
            }
        }
        
    def _protect_abbreviations(self, text: str, language: str = 'en') -> str:
        """
        Protect abbreviations from being split.
        
        Replace periods in abbreviations with temporary markers.
        """
        abbrevs = self.abbreviations.get(language, self.abbreviations['en'])
        
        for abbrev in abbrevs:
            # Create pattern for abbreviation followed by period
            pattern = r'\b' + re.escape(abbrev) + r'\.'
            # Replace with temporary marker preserving case
            text = re.sub(pattern, lambda m: m.group(0).replace('.', '⟨DOT⟩'), text, flags=re.IGNORECASE)
            
        return text
        
    def _restore_abbreviations(self, text: str) -> str:
        """Restore protected abbreviations"""
        return text.replace('⟨DOT⟩', '.')
        
    def _split_by_regex(self, text: str, language: str = 'en') -> List[str]:
        """
        Split text into sentences using regex patterns.
        
        Args:
            text: Text to split
            language: Language code for language-specific handling
            
        Returns:
            List of sentences
        """
        if not text.strip():
            return []
            
        # Protect abbreviations
        protected_text = self._protect_abbreviations(text, language)
        
        # Handle different splitting strategies
        sentences = []
        
        # First, try splitting on double line breaks (paragraph breaks)
        paragraphs = self.line_break_pattern.split(protected_text)
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Split paragraph into sentences
            para_sentences = self._split_paragraph(paragraph, language)
            sentences.extend(para_sentences)
            
        # Restore abbreviations
        sentences = [self._restore_abbreviations(s) for s in sentences]
        
        return sentences
        
    def _split_paragraph(self, paragraph: str, language: str) -> List[str]:
        """Split a single paragraph into sentences"""
        
        # Simple but robust approach for review text
        # Handle specific patterns that are common in reviews
        
        # For very short text, don't split
        if len(paragraph.strip()) <= 50:
            return [paragraph.strip()] if paragraph.strip() else []
            
        # For emoji-heavy text, be more conservative
        emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001F900-\U0001F9FF\U00002600-\U000026FF]'
        emoji_count = len(re.findall(emoji_pattern, paragraph))
        
        # If more than 25% emojis, treat as single sentence
        if emoji_count > len(paragraph) * 0.25:
            return [paragraph.strip()]
            
        # Standard sentence splitting with better space preservation
        # Split on sentence terminators followed by whitespace and capital letter
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)
        
        # If no capital letter splits, try any whitespace split
        if len(parts) <= 1:
            parts = re.split(r'(?<=[.!?])\s+', paragraph)
            
        if len(parts) <= 1:
            return [paragraph.strip()] if paragraph.strip() else []
            
        # Clean and validate parts
        sentences = []
        for part in parts:
            part = part.strip()
            if part and len(part) >= self.min_sentence_length:
                sentences.append(part)
            elif part and len(sentences) > 0:
                # Merge short part with previous sentence
                sentences[-1] = sentences[-1] + ' ' + part
            elif part:
                # Keep standalone short parts
                sentences.append(part)
                
        return sentences if sentences else [paragraph.strip()]
        
    def _split_by_heuristics(self, text: str, language: str) -> List[str]:
        """
        Split text using heuristics when regex splitting fails.
        
        This handles cases like:
        - Very short reviews
        - Reviews with only emojis
        - Informal punctuation
        """
        text = text.strip()
        
        if not text:
            return []
            
        # If text is very short, treat as single sentence
        if len(text) <= 50:
            return [text]
            
        # Try splitting on common review patterns
        sentences = []
        
        # Split on emoji sequences (if they seem to separate thoughts)
        if self.preserve_emojis:
            # Look for emoji followed by text
            emoji_split = re.split(r'([^\w\s]+)(?=\s*[A-ZА-ЯЁa-zа-яё])', text)
            
            current = ""
            for part in emoji_split:
                part = part.strip()
                if not part:
                    continue
                    
                if self.emoji_pattern.search(part):
                    # This is an emoji or punctuation
                    current += part + " "
                else:
                    # This is text
                    current += part
                    
                    # Check if we should split here
                    if len(current.strip()) > 30 and any(p in current for p in ['.', '!', '?']):
                        sentences.append(current.strip())
                        current = ""
                        
            if current.strip():
                sentences.append(current.strip())
                
        else:
            sentences = [text]
            
        return sentences if sentences else [text]
        
    def split(self, text: str, language: str = 'en') -> SentenceSplitResult:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            language: Language code for language-specific handling
            
        Returns:
            SentenceSplitResult with sentences and metadata
        """
        if not text or not isinstance(text, str):
            return SentenceSplitResult(
                sentences=[],
                original_text=text or "",
                method="empty"
            )
            
        logger.debug(f"Splitting text into sentences: {text[:100]}...")
        
        # Use regex-based splitting
        sentences = self._split_by_regex(text, language)
        
        # Post-processing: filter and clean sentences
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Filter by length
            if len(sentence) < self.min_sentence_length:
                continue
                
            # Truncate if too long
            if len(sentence) > self.max_sentence_length:
                sentence = sentence[:self.max_sentence_length].strip()
                
            # Skip if only punctuation or whitespace
            if re.match(r'^[\s\W]*$', sentence):
                continue
                
            cleaned_sentences.append(sentence)
            
        # If no sentences found, use original text as single sentence
        if not cleaned_sentences and text.strip():
            cleaned_sentences = [text.strip()]
            
        result = SentenceSplitResult(
            sentences=cleaned_sentences,
            original_text=text,
            method="regex"
        )
        
        logger.debug(f"Split into {result.sentence_count} sentences, avg length: {result.avg_sentence_length:.1f}")
        
        return result
        
    def batch_split(self, texts: List[str], languages: Optional[List[str]] = None) -> List[SentenceSplitResult]:
        """
        Split multiple texts into sentences.
        
        Args:
            texts: List of texts to split
            languages: Optional list of language codes (same length as texts)
            
        Returns:
            List of SentenceSplitResults
        """
        if languages is None:
            languages = ['en'] * len(texts)
        elif len(languages) != len(texts):
            logger.warning("Languages list length doesn't match texts list, using 'en' for all")
            languages = ['en'] * len(texts)
            
        results = []
        for text, language in zip(texts, languages):
            result = self.split(text, language)
            results.append(result)
            
        return results
        
    def get_statistics(self, results: List[SentenceSplitResult]) -> Dict:
        """
        Get statistics from sentence splitting results.
        
        Args:
            results: List of SentenceSplitResults
            
        Returns:
            Dictionary with splitting statistics
        """
        total_texts = len(results)
        total_sentences = sum(r.sentence_count for r in results)
        all_sentence_lengths = []
        
        for result in results:
            all_sentence_lengths.extend([len(s) for s in result.sentences])
            
        return {
            "total_texts": total_texts,
            "total_sentences": total_sentences,
            "avg_sentences_per_text": total_sentences / total_texts if total_texts > 0 else 0,
            "avg_sentence_length": sum(all_sentence_lengths) / len(all_sentence_lengths) if all_sentence_lengths else 0,
            "min_sentence_length": min(all_sentence_lengths) if all_sentence_lengths else 0,
            "max_sentence_length": max(all_sentence_lengths) if all_sentence_lengths else 0,
            "single_sentence_texts": len([r for r in results if r.sentence_count == 1]),
            "multi_sentence_texts": len([r for r in results if r.sentence_count > 1])
        }