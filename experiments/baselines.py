"""
ICML Baselines Module (P0.5)
============================

Strong baselines to rule out confounds:
- "It's just shorter prompts"
- "Any editing hurts"

Baselines:
1. Original (no compression)
2. Truncate-to-budget (first N tokens) - VERY IMPORTANT
3. Truncate-last (last N tokens)
4. Random deletion to same budget
5. Structure-preserving (keep headings/bullets)
"""

import random
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from transformers import AutoTokenizer
import numpy as np


@dataclass
class CompressionResult:
    """Standard result format for all compressors/baselines."""
    original: str
    compressed: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    method: str
    metadata: Dict = None


class BaseCompressor(ABC):
    """Abstract base class for all compression methods."""
    
    def __init__(self, tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.name = "base"
    
    @abstractmethod
    def compress(self, text: str, target_ratio: float) -> CompressionResult:
        """Compress text to target ratio."""
        pass
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens using downstream tokenizer."""
        return len(self.tokenizer.encode(text))
    
    def _tokens_to_text(self, text: str, n_tokens: int) -> str:
        """Get text corresponding to first n tokens."""
        tokens = self.tokenizer.encode(text)
        truncated = tokens[:n_tokens]
        return self.tokenizer.decode(truncated, skip_special_tokens=True)


# =============================================================================
# Baseline 1: Original (No Compression)
# =============================================================================

class OriginalBaseline(BaseCompressor):
    """No compression - returns original text."""
    
    def __init__(self, tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        super().__init__(tokenizer_name)
        self.name = "original"
    
    def compress(self, text: str, target_ratio: float = 1.0) -> CompressionResult:
        tokens = self._count_tokens(text)
        return CompressionResult(
            original=text,
            compressed=text,
            original_tokens=tokens,
            compressed_tokens=tokens,
            compression_ratio=1.0,
            method=self.name
        )


# =============================================================================
# Baseline 2: Truncate First N Tokens (CRITICAL BASELINE)
# =============================================================================

class TruncateFirstBaseline(BaseCompressor):
    """
    Keep first N tokens to match target ratio.
    
    This is the most important baseline - if your method doesn't beat this,
    reviewers will question whether learned compression adds value.
    """
    
    def __init__(self, tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        super().__init__(tokenizer_name)
        self.name = "truncate_first"
    
    def compress(self, text: str, target_ratio: float) -> CompressionResult:
        original_tokens = self._count_tokens(text)
        target_tokens = max(1, int(original_tokens * target_ratio))
        
        compressed = self._tokens_to_text(text, target_tokens)
        compressed_tokens = self._count_tokens(compressed)
        
        return CompressionResult(
            original=text,
            compressed=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            method=self.name,
            metadata={'target_ratio': target_ratio}
        )


# =============================================================================
# Baseline 3: Truncate Last N Tokens
# =============================================================================

class TruncateLastBaseline(BaseCompressor):
    """Keep last N tokens to match target ratio."""
    
    def __init__(self, tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        super().__init__(tokenizer_name)
        self.name = "truncate_last"
    
    def compress(self, text: str, target_ratio: float) -> CompressionResult:
        original_tokens = self._count_tokens(text)
        target_tokens = max(1, int(original_tokens * target_ratio))
        
        # Get last N tokens
        tokens = self.tokenizer.encode(text)
        truncated = tokens[-target_tokens:]
        compressed = self.tokenizer.decode(truncated, skip_special_tokens=True)
        compressed_tokens = self._count_tokens(compressed)
        
        return CompressionResult(
            original=text,
            compressed=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            method=self.name,
            metadata={'target_ratio': target_ratio}
        )


# =============================================================================
# Baseline 4: Random Deletion
# =============================================================================

class RandomDeletionBaseline(BaseCompressor):
    """
    Randomly delete tokens to reach target budget.
    
    This baseline tests whether the learned importance scoring
    is better than random selection.
    """
    
    def __init__(self, tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct", seed: int = 42):
        super().__init__(tokenizer_name)
        self.name = "random_deletion"
        self.seed = seed
    
    def compress(self, text: str, target_ratio: float) -> CompressionResult:
        random.seed(self.seed)
        
        original_tokens = self._count_tokens(text)
        target_tokens = max(1, int(original_tokens * target_ratio))
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= target_tokens:
            compressed = text
        else:
            # Randomly select tokens to keep
            keep_indices = sorted(random.sample(range(len(tokens)), target_tokens))
            kept_tokens = [tokens[i] for i in keep_indices]
            compressed = self.tokenizer.decode(kept_tokens, skip_special_tokens=True)
        
        compressed_tokens = self._count_tokens(compressed)
        
        return CompressionResult(
            original=text,
            compressed=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            method=self.name,
            metadata={'target_ratio': target_ratio, 'seed': self.seed}
        )


# =============================================================================
# Baseline 5: Structure-Preserving
# =============================================================================

class StructurePreservingBaseline(BaseCompressor):
    """
    Keep structural elements (headings, bullets, numbered lists),
    delete body content.
    
    Tests whether structure alone captures constraints.
    """
    
    def __init__(self, tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        super().__init__(tokenizer_name)
        self.name = "structure_preserving"
        
        # Patterns for structural elements
        self.structure_patterns = [
            r'^#{1,6}\s+.*$',          # Markdown headers
            r'^\*\s+.*$',               # Bullet points
            r'^-\s+.*$',                # Dash bullets
            r'^\d+\.\s+.*$',            # Numbered lists
            r'^[A-Z][^.!?]*[.!?]$',     # Single sentence (likely important)
            r'^\*\*.*\*\*$',            # Bold text
            r'^<[^>]+>.*</[^>]+>$',     # XML-style tags
        ]
    
    def _is_structural(self, line: str) -> bool:
        """Check if line is a structural element."""
        line = line.strip()
        if not line:
            return False
        
        for pattern in self.structure_patterns:
            if re.match(pattern, line, re.MULTILINE):
                return True
        
        # Also keep very short lines (likely headings/labels)
        if len(line) < 50 and line.endswith(':'):
            return True
        
        return False
    
    def compress(self, text: str, target_ratio: float) -> CompressionResult:
        original_tokens = self._count_tokens(text)
        target_tokens = max(1, int(original_tokens * target_ratio))
        
        # Split into lines and identify structural elements
        lines = text.split('\n')
        structural_lines = []
        body_lines = []
        
        for line in lines:
            if self._is_structural(line):
                structural_lines.append(line)
            else:
                body_lines.append(line)
        
        # Start with all structural lines
        compressed_parts = structural_lines.copy()
        current_tokens = self._count_tokens('\n'.join(compressed_parts))
        
        # Add body lines until we hit target
        for line in body_lines:
            if not line.strip():
                continue
            
            test_text = '\n'.join(compressed_parts + [line])
            test_tokens = self._count_tokens(test_text)
            
            if test_tokens <= target_tokens:
                compressed_parts.append(line)
                current_tokens = test_tokens
            else:
                break
        
        compressed = '\n'.join(compressed_parts)
        compressed_tokens = self._count_tokens(compressed)
        
        return CompressionResult(
            original=text,
            compressed=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            method=self.name,
            metadata={
                'target_ratio': target_ratio,
                'structural_lines': len(structural_lines),
                'body_lines': len(body_lines)
            }
        )


# =============================================================================
# Baseline Manager
# =============================================================================

class BaselineManager:
    """
    Manage all baselines for consistent evaluation.
    """
    
    def __init__(self, tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.baselines = {
            'original': OriginalBaseline(tokenizer_name),
            'truncate_first': TruncateFirstBaseline(tokenizer_name),
            'truncate_last': TruncateLastBaseline(tokenizer_name),
            'random_deletion': RandomDeletionBaseline(tokenizer_name),
            'structure_preserving': StructurePreservingBaseline(tokenizer_name),
        }
    
    def run_all(
        self, 
        text: str, 
        target_ratio: float
    ) -> Dict[str, CompressionResult]:
        """Run all baselines on a single text."""
        results = {}
        for name, baseline in self.baselines.items():
            results[name] = baseline.compress(text, target_ratio)
        return results
    
    def batch_run_all(
        self,
        texts: List[str],
        target_ratio: float
    ) -> Dict[str, List[CompressionResult]]:
        """Run all baselines on multiple texts."""
        results = {name: [] for name in self.baselines}
        
        for text in texts:
            for name, baseline in self.baselines.items():
                results[name].append(baseline.compress(text, target_ratio))
        
        return results
    
    def get_baseline(self, name: str) -> BaseCompressor:
        """Get a specific baseline."""
        return self.baselines.get(name)
    
    def compare_with_method(
        self,
        texts: List[str],
        method_results: List[CompressionResult],
        target_ratio: float
    ) -> Dict:
        """
        Compare a compression method against all baselines.
        
        Returns comparison statistics showing where method differs.
        """
        baseline_results = self.batch_run_all(texts, target_ratio)
        
        comparisons = {}
        
        for baseline_name, baseline_res in baseline_results.items():
            # Compare token counts
            method_tokens = [r.compressed_tokens for r in method_results]
            baseline_tokens = [r.compressed_tokens for r in baseline_res]
            
            # Compare lengths
            method_lens = [len(r.compressed) for r in method_results]
            baseline_lens = [len(r.compressed) for r in baseline_res]
            
            comparisons[baseline_name] = {
                'method_avg_tokens': np.mean(method_tokens),
                'baseline_avg_tokens': np.mean(baseline_tokens),
                'method_avg_len': np.mean(method_lens),
                'baseline_avg_len': np.mean(baseline_lens),
                'token_diff': np.mean(method_tokens) - np.mean(baseline_tokens),
            }
        
        return comparisons

