"""
AutoCompressor Wrapper
======================

Soft compression using princeton-nlp/AutoCompressor.

AutoCompressor compresses context into summary vectors (soft prompts)
that can be used to condition model generation.

Paper: https://arxiv.org/abs/2305.14788
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch

from .base import BaseCompressor, CompressionResult


class AutoCompressorWrapper(BaseCompressor):
    """
    AutoCompressor wrapper for soft compression.
    
    Usage:
        compressor = AutoCompressorWrapper()
        result = compressor.compress("Your text here")
        
        # Access embeddings
        embeddings = result.embeddings  # Shape: [50, 4096]
    """
    
    MODEL_NAME = "princeton-nlp/AutoCompressor-Llama-2-7b-6k"
    NUM_SUMMARY_TOKENS = 50  # Fixed by the model
    
    def __init__(
        self,
        model_name: str = None,
        device: str = "cuda",
        autocompressor_path: Optional[str] = None
    ):
        """
        Initialize AutoCompressor.
        
        Args:
            model_name: Model name (default: princeton-nlp/AutoCompressor-Llama-2-7b-6k)
            device: Device to use
            autocompressor_path: Path to AutoCompressor repo (for importing)
        """
        super().__init__(device)
        
        self.model_name = model_name or self.MODEL_NAME
        self.num_summary_tokens = self.NUM_SUMMARY_TOKENS
        
        # Add AutoCompressor to path if provided
        if autocompressor_path:
            sys.path.insert(0, str(autocompressor_path))
        
        self._load_model()
    
    @property
    def name(self) -> str:
        return "autocompressor"
    
    @property
    def is_soft(self) -> bool:
        return True
    
    def _load_model(self):
        """Load AutoCompressor model."""
        from transformers import AutoTokenizer
        
        logging.info(f"Loading AutoCompressor: {self.model_name}")
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Try to load the real AutoCompressor model
        try:
            from auto_compressor import LlamaAutoCompressorModel
            
            self._model = LlamaAutoCompressorModel.from_pretrained(
                self.model_name,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            self._model.eval()
            self._is_real_model = True
            
            logging.info(f"AutoCompressor loaded (hidden_size={self._model.config.hidden_size})")
            
        except ImportError as e:
            logging.error(f"Failed to import AutoCompressor: {e}")
            logging.error("Please ensure AutoCompressor repo is in PYTHONPATH")
            raise
    
    def compress(self, text: str, rate: float = None) -> CompressionResult:
        """
        Compress text into summary embeddings.
        
        Args:
            text: Input text
            rate: Ignored for soft compression (fixed 50 summary tokens)
            
        Returns:
            CompressionResult with embeddings
        """
        if not text or len(text.strip()) < 10:
            # Return zero embeddings for empty input
            embeddings = torch.zeros(
                self.num_summary_tokens,
                self._model.config.hidden_size
            )
            return CompressionResult(
                original_text=text,
                original_tokens=0,
                compressed_text="",
                compressed_tokens=self.num_summary_tokens,
                embeddings=embeddings,
                compression_ratio=0.0,
                method=self.name
            )
        
        start_time = time.time()
        
        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=6144,
            add_special_tokens=False
        ).to(self.device)
        
        original_tokens = inputs.input_ids.shape[1]
        
        # Get summary embeddings
        with torch.no_grad():
            output = self._model(inputs.input_ids, output_softprompt=True)
            # output.softprompt shape: [1, num_summary_tokens, hidden_size]
            embeddings = output.softprompt.squeeze(0).cpu()
            # embeddings shape: [num_summary_tokens, hidden_size]
        
        compression_time = time.time() - start_time
        
        return CompressionResult(
            original_text=text,
            original_tokens=original_tokens,
            compressed_text="",  # No text for soft compression
            compressed_tokens=self.num_summary_tokens,
            embeddings=embeddings,
            compression_ratio=self.num_summary_tokens / original_tokens if original_tokens > 0 else 0.0,
            method=self.name,
            compression_time=compression_time
        )
    
    def generate_with_softprompt(
        self,
        softprompt: torch.Tensor,
        query: str,
        max_new_tokens: int = 256
    ) -> str:
        """
        Generate text using a soft prompt.
        
        Args:
            softprompt: Summary embeddings from compress()
            query: Query text
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        # Ensure correct shape: [1, num_summary_tokens, hidden_size]
        if softprompt.dim() == 2:
            softprompt = softprompt.unsqueeze(0)
        
        softprompt = softprompt.to(self.device)
        if softprompt.dtype != self._model.dtype:
            softprompt = softprompt.to(self._model.dtype)
        
        # Tokenize query
        query_ids = self._tokenizer.encode(
            query,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            # Manual generation since .generate() may not work
            generated_ids = query_ids.clone()
            
            for _ in range(max_new_tokens):
                outputs = self._model(
                    input_ids=generated_ids,
                    softprompt=softprompt
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                
                if next_token_id.item() == self._tokenizer.eos_token_id:
                    break
            
            # Decode only generated part
            new_tokens = generated_ids[0, query_ids.shape[1]:]
            response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response




