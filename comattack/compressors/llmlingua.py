"""LLMLingua compressor."""

import logging
from llmlingua import PromptCompressor


class LLMLinguaCompressor:
    """Compressor using LLMLingua or LLMLingua2."""
    
    def __init__(self, version: int = 2, device: str = "cuda"):
        self.version = version
        self.device = device
        self._compressor = None
        self._load_model()
    
    def _load_model(self):
        """Load LLMLingua model."""
        if self.version == 2:
            self._compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True
            )
        else:
            self._compressor = PromptCompressor("NousResearch/Llama-2-7b-hf")
        
        logging.info(f"LLMLingua v{self.version} loaded")
    
    def compress(self, text: str, rate: float = 0.6):
        """Compress text. Input must be a single prompt string."""
        text = str(text).strip()
        
        if not text or len(text) < 10:
            return {
                "compressed_text": text,
                "original_tokens": 0,
                "compressed_tokens": 0,
                "compression_ratio": 1.0
            }
        
        try:
            result = self._compressor.compress_prompt(text, rate=rate)
            
            # Handle case where result might be a list (error case)
            if isinstance(result, list):
                logging.warning(f"Compression returned list instead of dict, text may be too long")
                raise ValueError("Input too long for compression")
            
            return {
                "compressed_text": result.get("compressed_prompt", text),
                "original_tokens": result.get("origin_tokens", len(text.split())),
                "compressed_tokens": result.get("compressed_tokens", len(result.get("compressed_prompt", text).split())),
                "compression_ratio": result.get("compressed_tokens", 0) / result.get("origin_tokens", 1) if result.get("origin_tokens", 0) > 0 else 1.0
            }
        except Exception as e:
            logging.warning(f"Compression failed: {e}")
            # Re-raise if it's a length issue, otherwise return original
            if "too long" in str(e).lower() or "get_seq_length" in str(e):
                raise
            return {
                "compressed_text": text,
                "original_tokens": len(text.split()),
                "compressed_tokens": len(text.split()),
                "compression_ratio": 1.0
            }
    
    def cleanup(self):
        """Release model resources."""
        if self._compressor:
            del self._compressor
            self._compressor = None
