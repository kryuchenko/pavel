"""
Content deduplication for Google Play reviews.

Uses multiple techniques to identify and handle duplicate/near-duplicate content:
- Exact text matching
- Fuzzy string matching
- SimHash for semantic similarity
- MinHash for set similarity
"""

import hashlib
import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

try:
    from difflib import SequenceMatcher
    DIFFLIB_AVAILABLE = True
except ImportError:
    DIFFLIB_AVAILABLE = False

from pavel.core.logger import get_logger

logger = get_logger(__name__)

@dataclass
class DuplicateGroup:
    """Group of duplicate/similar content"""
    canonical_text: str  # Representative text for the group
    duplicate_texts: List[str]  # All texts in the group
    similarity_scores: List[float]  # Similarity scores to canonical
    group_id: str  # Unique identifier for the group
    method: str  # Method used to detect duplicates

@dataclass
class DeduplicationResult:
    """Result of deduplication process"""
    original_count: int
    unique_count: int
    duplicate_count: int
    duplicate_groups: List[DuplicateGroup]
    similarity_threshold: float
    method: str

class ContentDeduplicator:
    """
    Multi-method content deduplicator for app reviews.
    
    Features:
    - Exact string matching
    - Fuzzy string matching with configurable threshold
    - Length-based filtering
    - Normalization before comparison
    - Grouping of similar content
    """
    
    def __init__(self,
                 similarity_threshold: float = 0.8,
                 min_length_for_comparison: int = 10,
                 normalize_for_comparison: bool = True):
        self.similarity_threshold = similarity_threshold
        self.min_length_for_comparison = min_length_for_comparison
        self.normalize_for_comparison = normalize_for_comparison
        
        self._setup_normalization_patterns()
        
    def _setup_normalization_patterns(self):
        """Set up patterns for text normalization during comparison"""
        
        # Patterns to normalize text for better duplicate detection
        self.normalization_patterns = [
            (re.compile(r'\s+'), ' '),  # Multiple whitespace to single space
            (re.compile(r'[.!?]+'), '.'),  # Multiple punctuation to single
            (re.compile(r'[^\w\s.]', re.UNICODE), ''),  # Remove non-alphanumeric except spaces and periods
        ]
        
    def _normalize_for_comparison(self, text: str) -> str:
        """
        Normalize text for comparison purposes.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not self.normalize_for_comparison:
            return text.strip().lower()
            
        normalized = text.strip().lower()
        
        # Apply normalization patterns
        for pattern, replacement in self.normalization_patterns:
            normalized = pattern.sub(replacement, normalized)
            
        return normalized.strip()
        
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not DIFFLIB_AVAILABLE:
            # Fallback to simple string comparison
            return 1.0 if text1 == text2 else 0.0
            
        # Normalize texts for comparison
        norm1 = self._normalize_for_comparison(text1)
        norm2 = self._normalize_for_comparison(text2)
        
        # Use SequenceMatcher for fuzzy similarity
        return SequenceMatcher(None, norm1, norm2).ratio()
        
    def _create_text_signature(self, text: str) -> str:
        """
        Create a signature for text that can be used for quick comparison.
        
        Args:
            text: Text to create signature for
            
        Returns:
            Text signature (hash)
        """
        normalized = self._normalize_for_comparison(text)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
        
    def find_exact_duplicates(self, texts: List[str]) -> Dict[str, List[int]]:
        """
        Find exact duplicates using text signatures.
        
        Args:
            texts: List of texts to check for duplicates
            
        Returns:
            Dictionary mapping signatures to lists of indices
        """
        signature_to_indices = defaultdict(list)
        
        for i, text in enumerate(texts):
            if len(text) >= self.min_length_for_comparison:
                signature = self._create_text_signature(text)
                signature_to_indices[signature].append(i)
                
        # Keep only signatures with multiple texts
        duplicates = {sig: indices for sig, indices in signature_to_indices.items() 
                     if len(indices) > 1}
        
        return duplicates
        
    def find_fuzzy_duplicates(self, texts: List[str]) -> List[DuplicateGroup]:
        """
        Find fuzzy duplicates using similarity comparison.
        
        Args:
            texts: List of texts to check for duplicates
            
        Returns:
            List of DuplicateGroups
        """
        if not DIFFLIB_AVAILABLE:
            logger.warning("difflib not available, skipping fuzzy deduplication")
            return []
            
        duplicate_groups = []
        processed_indices = set()
        
        for i, text1 in enumerate(texts):
            if i in processed_indices or len(text1) < self.min_length_for_comparison:
                continue
                
            # Find all similar texts to text1
            similar_indices = [i]  # Include the text itself
            similar_texts = [text1]
            similarity_scores = [1.0]  # Perfect similarity to itself
            
            for j, text2 in enumerate(texts[i+1:], i+1):
                if j in processed_indices or len(text2) < self.min_length_for_comparison:
                    continue
                    
                similarity = self._calculate_similarity(text1, text2)
                
                if similarity >= self.similarity_threshold:
                    similar_indices.append(j)
                    similar_texts.append(text2)
                    similarity_scores.append(similarity)
                    
            # If we found similar texts, create a duplicate group
            if len(similar_indices) > 1:
                # Mark all these indices as processed
                processed_indices.update(similar_indices)
                
                # Create duplicate group
                group_id = f"fuzzy_{len(duplicate_groups)}"
                group = DuplicateGroup(
                    canonical_text=text1,  # Use first text as canonical
                    duplicate_texts=similar_texts,
                    similarity_scores=similarity_scores,
                    group_id=group_id,
                    method="fuzzy"
                )
                
                duplicate_groups.append(group)
                
        return duplicate_groups
        
    def deduplicate(self, texts: List[str], method: str = "both") -> DeduplicationResult:
        """
        Perform deduplication on a list of texts.
        
        Args:
            texts: List of texts to deduplicate
            method: Method to use ("exact", "fuzzy", "both")
            
        Returns:
            DeduplicationResult with details about the process
        """
        original_count = len(texts)
        duplicate_groups = []
        
        logger.info(f"Starting deduplication of {original_count} texts using method: {method}")
        
        # Find exact duplicates
        if method in ["exact", "both"]:
            exact_duplicates = self.find_exact_duplicates(texts)
            
            for signature, indices in exact_duplicates.items():
                if len(indices) > 1:
                    # Create duplicate group for exact matches
                    canonical_text = texts[indices[0]]
                    duplicate_texts = [texts[i] for i in indices]
                    
                    group_id = f"exact_{len(duplicate_groups)}"
                    group = DuplicateGroup(
                        canonical_text=canonical_text,
                        duplicate_texts=duplicate_texts,
                        similarity_scores=[1.0] * len(duplicate_texts),  # All exact matches
                        group_id=group_id,
                        method="exact"
                    )
                    
                    duplicate_groups.append(group)
                    
        # Find fuzzy duplicates (only if not already found as exact)
        if method in ["fuzzy", "both"]:
            # Get indices that are not part of exact duplicate groups
            exact_indices = set()
            for group in duplicate_groups:
                if group.method == "exact":
                    for i, text in enumerate(texts):
                        if text in group.duplicate_texts:
                            exact_indices.add(i)
                            
            # Create list of texts not in exact duplicates for fuzzy matching
            remaining_texts = [text for i, text in enumerate(texts) if i not in exact_indices]
            
            if remaining_texts:
                fuzzy_groups = self.find_fuzzy_duplicates(remaining_texts)
                duplicate_groups.extend(fuzzy_groups)
                
        # Calculate unique count
        duplicate_count = sum(len(group.duplicate_texts) - 1 for group in duplicate_groups)
        unique_count = original_count - duplicate_count
        
        result = DeduplicationResult(
            original_count=original_count,
            unique_count=unique_count,
            duplicate_count=duplicate_count,
            duplicate_groups=duplicate_groups,
            similarity_threshold=self.similarity_threshold,
            method=method
        )
        
        logger.info(f"Deduplication complete: {unique_count} unique, {duplicate_count} duplicates, "
                   f"{len(duplicate_groups)} groups")
        
        return result
        
    def get_unique_texts(self, texts: List[str], method: str = "both") -> List[str]:
        """
        Get list of unique texts after deduplication.
        
        Args:
            texts: List of texts to deduplicate
            method: Method to use ("exact", "fuzzy", "both")
            
        Returns:
            List of unique texts (canonical texts from each group)
        """
        result = self.deduplicate(texts, method)
        
        # Create set of texts to remove (non-canonical duplicates)
        texts_to_remove = set()
        canonical_texts = set()
        
        for group in result.duplicate_groups:
            # Keep first text as canonical
            canonical_texts.add(group.canonical_text)
            # Mark other texts for removal
            for text in group.duplicate_texts[1:]:  # Skip first (canonical)
                texts_to_remove.add(text)
                
        # Build final list keeping order but removing duplicates
        final_unique_texts = []
        seen = set()
        
        for text in texts:
            if text not in texts_to_remove and text not in seen:
                final_unique_texts.append(text)
                seen.add(text)
                
        return final_unique_texts
        
    def analyze_duplication_patterns(self, texts: List[str]) -> Dict:
        """
        Analyze patterns in duplicate content.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with analysis results
        """
        result = self.deduplicate(texts, method="both")
        
        # Analyze group sizes
        group_sizes = [len(group.duplicate_texts) for group in result.duplicate_groups]
        
        # Analyze similarity scores
        all_similarities = []
        for group in result.duplicate_groups:
            if group.method == "fuzzy":
                all_similarities.extend(group.similarity_scores)
                
        analysis = {
            "total_texts": len(texts),
            "unique_texts": result.unique_count,
            "duplicate_groups": len(result.duplicate_groups),
            "duplication_rate": result.duplicate_count / len(texts) if texts else 0,
            "avg_group_size": sum(group_sizes) / len(group_sizes) if group_sizes else 0,
            "max_group_size": max(group_sizes) if group_sizes else 0,
            "exact_groups": len([g for g in result.duplicate_groups if g.method == "exact"]),
            "fuzzy_groups": len([g for g in result.duplicate_groups if g.method == "fuzzy"]),
            "avg_similarity": sum(all_similarities) / len(all_similarities) if all_similarities else 0,
            "similarity_threshold": self.similarity_threshold
        }
        
        return analysis