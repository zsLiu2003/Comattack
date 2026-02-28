"""
Base Compressor
===============

Abstract base class for all compression methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Any
import torch


@dataclass
class CompressionResult:
    """
    Result of compression operation.
    
    For hard compression: compressed_text contains readable text
    For soft compression: embeddings contains tensor, compressed_text is empty
    """
    # Original
    original_text: str
    original_tokens: int
    
    # Compressed
    compressed_text: str  # For hard compression
    compressed_tokens: int
    embeddings: Optional[torch.Tensor] = None  # For soft compression
    
    # Metadata
    compression_ratio: float = 0.0
    method: str = ""
    rate: float = 0.0
    compression_time: float = 0.0
    
    @property
    def is_soft(self) -> bool:
        """Whether this is a soft compression result."""
        return self.embeddings is not None
    
    def __repr__(self):
        if self.is_soft:
            return f"CompressionResult(soft, ratio={self.compression_ratio:.2f}, shape={self.embeddings.shape})"
        else:
            return f"CompressionResult(hard, ratio={self.compression_ratio:.2f}, tokens={self.original_tokens}→{self.compressed_tokens})"


class BaseCompressor(ABC):
    """
    Abstract base class for compressors.
    
    All compressors must implement:
    - compress(): Compress a single text
    - name: Property returning compressor name
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize compressor.
        
        Args:
            device: Device to use ("cuda" or "cpu")
        """
        self.device = device
        self._model = None
        self._tokenizer = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return compressor name."""
        pass
    
    @property
    def is_soft(self) -> bool:
        """Whether this compressor produces soft (embedding) outputs."""
        return False
    
    @abstractmethod
    def compress(self, text: str, rate: float = 0.5) -> CompressionResult:
        """
        Compress text.
        
        Args:
            text: Input text to compress
            rate: Target compression rate (0.5 = keep 50%)
            
        Returns:
            CompressionResult with compression details
        """
        pass
    
    def compress_batch(
        self, 
        texts: list, 
        rate: float = 0.5,
        show_progress: bool = True
    ) -> list:
        """
        Compress multiple texts.
        
        Args:
            texts: List of texts to compress
            rate: Target compression rate
            show_progress: Show progress bar
            
        Returns:
            List of CompressionResult
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(texts, desc=f"{self.name}") if show_progress else texts
        
        for text in iterator:
            result = self.compress(text, rate=rate)
            results.append(result)
        
        return results
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if self._tokenizer is None:
            # Fallback to whitespace tokenization
            return len(text.split())
        return len(self._tokenizer.encode(text))
    
    def cleanup(self):
        """Release model resources."""
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        # Clear CUDA cache
        if self.device == "cuda":
            torch.cuda.empty_cache()




