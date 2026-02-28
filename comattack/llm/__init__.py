"""
LLM Provider Module
===================

Two modes:
- Server: Use utils/llm_provider.py (for remote API or vLLM server)
- Offline: Use vLLM to load HuggingFace models directly

Usage:
    from comattack.llm import create_llm, LLMConfig
    
    # Server mode (closed-source API)
    llm = create_llm(LLMConfig(model_name="gpt-5.2"))
    
    # Offline mode (vLLM loads HuggingFace)
    llm = create_llm(LLMConfig(model_name="Qwen/Qwen3-8B"))
    
    # Explicit mode selection
    llm = create_llm(LLMConfig(model_name="Qwen/Qwen3-8B", provider="server"))  # vLLM server
    llm = create_llm(LLMConfig(model_name="Qwen/Qwen3-8B", provider="offline")) # vLLM offline
"""

from .base import BaseLLM
from .factory import create_llm, LLMConfig
from .vllm_provider import VLLMServerLLM, VLLMOfflineLLM

# Optional: AutoCompressor (requires auto_compressor package)
try:
    from .autocompressor_provider import AutoCompressorLLM
    _HAS_AUTOCOMPRESSOR = True
except ImportError:
    AutoCompressorLLM = None
    _HAS_AUTOCOMPRESSOR = False

__all__ = [
    'BaseLLM',
    'create_llm',
    'LLMConfig',
    'VLLMServerLLM',
    'VLLMOfflineLLM',
    'AutoCompressorLLM',
]

