"""
LLM Factory - Create LLM instances based on configuration

Two modes:
- Server: Use existing utils/llm_provider.py (closed-source API or vLLM server)
- Offline: Use vLLM to load HuggingFace models directly

Auto-detection:
- Closed-source (gpt-5.2, gemini) -> Server mode
- Open-source (Qwen, Llama) -> Offline mode (or Server if vLLM server running)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path

from .base import BaseLLM

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """
    Configuration for LLM instantiation.
    
    Usage:
        # Server mode (closed-source API)
        LLMConfig(model_name="gpt-5.2", provider="server")
        
        # Server mode (vLLM server running locally)
        # First: vllm serve Qwen/Qwen3-8B --port 8000
        LLMConfig(model_name="Qwen/Qwen3-8B", provider="server")
        
        # Offline mode (vLLM loads HuggingFace model directly)
        LLMConfig(model_name="Qwen/Qwen3-8B", provider="offline")
        
        # Auto mode (default)
        # - Closed-source -> server
        # - Open-source -> offline
        LLMConfig(model_name="gpt-5.2")      # -> server
        LLMConfig(model_name="Qwen/Qwen3-8B") # -> offline
    """
    
    model_name: str
    provider: str = "auto"  # "server", "offline", "auto"
    model_type: str = "target"  # "target", "judge", "auxiliary"
    
    # vLLM Offline settings (for loading HuggingFace models)
    tensor_parallel_size: int = 1       # Number of GPUs
    gpu_memory_utilization: float = 0.9 # GPU memory fraction
    max_model_len: int = 8192           # Max context length
    quantization: Optional[str] = None  # "awq", "gptq", "squeezellm"
    trust_remote_code: bool = True
    
    # Extra kwargs
    extra: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Model Classification
# =============================================================================

# Closed-source models (MUST use API)
API_MODEL_PREFIXES = [
    "gpt-",           # OpenAI: gpt-3.5, gpt-4, gpt-4o, gpt-5, gpt-5.2
    "claude-",        # Anthropic: claude-3, claude-3.5-sonnet
    "gemini-",        # Google: gemini-pro, gemini-1.5
    "o1-",            # OpenAI reasoning: o1-preview, o1-mini
]

API_MODEL_EXACT = {
    "deepseek-chat", "deepseek-coder",  # DeepSeek API
    "moonshot-v1",                       # Moonshot
}

# Open-source models (use vLLM/HuggingFace)
# These are HuggingFace model paths
OPEN_SOURCE_PREFIXES = [
    "Qwen/",          # Qwen family
    "meta-llama/",    # Llama family
    "mistralai/",     # Mistral family
    "google/",        # Google open models (Gemma, etc.)
    "microsoft/",     # Microsoft (Phi, etc.)
    "THUDM/",         # ChatGLM
    "bigscience/",    # BLOOM
    "EleutherAI/",    # GPT-Neo, Pythia
    "princeton-nlp/", # AutoCompressor
    "facebook/",      # OPT, etc.
    "01-ai/",         # Yi
    "deepseek-ai/",   # DeepSeek open models
    "internlm/",      # InternLM
    "baichuan-inc/",  # Baichuan
]


def is_api_model(model_name: str) -> bool:
    """
    Check if model is closed-source and should use API.
    
    Examples:
        - gpt-4o-mini -> True (OpenAI)
        - gpt-5.2 -> True (OpenAI)
        - gemini-pro -> True (Google)
        - claude-3 -> True (Anthropic)
        - Qwen/Qwen3-8B -> False (open-source)
    """
    # Check exact match
    if model_name in API_MODEL_EXACT:
        return True
    
    # Check prefix match
    for prefix in API_MODEL_PREFIXES:
        if model_name.lower().startswith(prefix.lower()):
            return True
    
    return False


def is_local_model(model_name: str) -> bool:
    """
    Check if model is open-source and should use local inference (vLLM/HF).
    
    Examples:
        - Qwen/Qwen3-8B -> True
        - meta-llama/Llama-3.1-8B -> True
        - gpt-4 -> False (closed-source)
    """
    for prefix in OPEN_SOURCE_PREFIXES:
        if model_name.startswith(prefix):
            return True
    
    # Also check if it looks like a HuggingFace path (contains /)
    if "/" in model_name and not is_api_model(model_name):
        return True
    
    return False


def create_llm(config: LLMConfig) -> BaseLLM:
    """
    Create LLM instance based on configuration.
    
    All models driven by vLLM:
    
    Providers:
    - "server": vLLM OpenAI-compatible API
        - Closed-source: connects to remote API (base_url from config.yaml)
        - Open-source: connects to local vLLM server (need to start: vllm serve xxx)
    - "offline": vLLM directly loads HuggingFace model (open-source only)
    - "auto": Auto-detect:
        - Closed-source -> server (remote API)
        - Open-source -> offline (vLLM loads HuggingFace)
    
    Args:
        config: LLM configuration
        
    Returns:
        LLM instance (VLLMServerLLM or VLLMOfflineLLM)
    """
    model_name = config.model_name
    provider = config.provider
    
    # Auto-detect provider
    if provider == "auto":
        if is_api_model(model_name):
            # Closed-source -> server (remote API)
            provider = "server"
        elif is_local_model(model_name):
            # Open-source -> offline (vLLM loads HuggingFace)
            provider = "offline"
        else:
            # Unknown -> server
            provider = "server"
    
    logger.info(f"Creating LLM: {model_name}")
    logger.info(f"  Provider: {provider}")
    
    if provider == "server":
        # vLLM OpenAI-compatible API
        # - For closed-source: base_url = remote API (from config.yaml)
        # - For open-source: base_url = local vLLM server
        from .vllm_provider import VLLMServerLLM
        
        # Allow overriding base_url for local vLLM server
        base_url = config.extra.get("base_url")  # None means use config.yaml default
        
        return VLLMServerLLM(
            model_name=model_name,
            base_url=base_url,
            **{k: v for k, v in config.extra.items() if k != "base_url"}
        )
    
    elif provider == "offline":
        # vLLM directly loads HuggingFace model
        logger.info(f"  Loading HuggingFace model via vLLM...")
        logger.info(f"  Tensor Parallel: {config.tensor_parallel_size}")
        logger.info(f"  GPU Memory: {config.gpu_memory_utilization}")
        
        from .vllm_provider import VLLMOfflineLLM
        return VLLMOfflineLLM(
            model_name=model_name,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            trust_remote_code=config.trust_remote_code,
            quantization=config.quantization,
            **config.extra
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'server', 'offline', or 'auto'")


def create_llm_from_config(
    model_name: str,
    config_dict: Dict[str, Any]
) -> BaseLLM:
    """
    Convenience function to create LLM from config dictionary.
    
    Args:
        model_name: Model name/path
        config_dict: Configuration dictionary (from config.yaml)
        
    Returns:
        LLM instance
    """
    api_config = config_dict.get("api", {})
    
    llm_config = LLMConfig(
        model_name=model_name,
        api_key=api_config.get("api_keys", {}).get("openai", ""),
        base_url=api_config.get("base_url", "https://api2.aigcbest.top/v1"),
        device=config_dict.get("gpu", {}).get("default_device", "cuda")
    )
    
    return create_llm(llm_config)

