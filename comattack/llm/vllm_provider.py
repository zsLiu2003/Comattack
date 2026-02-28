"""
vLLM Provider - Unified LLM Interface
=====================================

All models driven by vLLM:

1. Server mode (VLLMServerLLM):
   - For closed-source models: Connect to remote API (gpt-5.2, gemini, claude)
   - For open-source models: Connect to local vLLM server
   - Both use vLLM's OpenAI-compatible API client

2. Offline mode (VLLMOfflineLLM):
   - For open-source models: vLLM directly loads HuggingFace models
   - No server needed, runs in process

Usage:
    # Server mode - closed-source (remote API)
    llm = VLLMServerLLM(
        model_name="gpt-5.2",
        base_url="https://api2.aigcbest.top/v1",
        api_key="sk-xxx"
    )
    
    # Server mode - open-source (local vLLM server)
    # First: vllm serve Qwen/Qwen3-8B --port 8000
    llm = VLLMServerLLM(
        model_name="Qwen/Qwen3-8B",
        base_url="http://localhost:8000/v1"
    )
    
    # Offline mode - open-source (vLLM loads HuggingFace directly)
    llm = VLLMOfflineLLM(model_name="Qwen/Qwen3-8B")
"""

import logging
import time
from typing import Optional, List, Dict, Any
from pathlib import Path

from .base import BaseLLM

logger = logging.getLogger(__name__)


def get_config():
    """Load config from config.yaml"""
    try:
        from comattack.utils import load_config
        return load_config()
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
        return {}


class VLLMServerLLM(BaseLLM):
    """
    Server mode - Use vLLM's OpenAI-compatible API client.
    
    Works for:
    - Closed-source models: base_url = remote API (e.g., https://api2.aigcbest.top/v1)
    - Open-source models: base_url = local vLLM server (e.g., http://localhost:8000/v1)
    """
    
    def __init__(
        self,
        model_name: str,
        base_url: str = None,  # Will read from config if None
        api_key: str = None,   # Will read from config if None
        max_retries: int = 3,
        timeout: int = 120,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        
        from openai import OpenAI, APIError, RateLimitError, APITimeoutError
        
        # Load config
        config = get_config()
        api_config = config.get("api", {})
        
        # Get base_url from config if not provided
        if base_url is None:
            base_url = api_config.get("base_url", "https://api2.aigcbest.top/v1")
        
        # Get api_key from config if not provided
        if api_key is None:
            api_key = api_config.get("api_keys", {}).get("openai", "")
        
        self.base_url = base_url
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Create OpenAI client (vLLM uses same interface)
        self.client = OpenAI(
            api_key=api_key if api_key else "EMPTY",
            base_url=base_url,
            timeout=timeout
        )
        
        # Store exception classes for retry logic
        self._APIError = APIError
        self._RateLimitError = RateLimitError
        self._APITimeoutError = APITimeoutError
        
        logger.info(f"VLLMServerLLM initialized: {model_name}")
        logger.info(f"  Base URL: {base_url}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        enable_thinking: bool = True,  # Qwen3 thinking mode
        **kwargs
    ) -> str:
        """Generate using vLLM OpenAI-compatible API."""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare extra_body for Qwen3 thinking mode
        # extra_body = {}
        # if "Qwen3" in self.model_name or "qwen3" in self.model_name.lower():
        #     extra_body["chat_template_kwargs"] = {
        #         "enable_thinking": enable_thinking
        #     }
        
        for attempt in range(self.max_retries):
            try:
                request_kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature if temperature > 0 else 0.01,
                    "max_tokens": max_tokens,
                }
                # if extra_body:
                #     request_kwargs["extra_body"] = extra_body
                
                response = self.client.chat.completions.create(**request_kwargs)
                
                content = response.choices[0].message.content
                return content.strip() if content else ""
                
            except self._RateLimitError as e:
                wait = 5 * (attempt + 1)
                logger.warning(f"Rate limit (attempt {attempt + 1}), waiting {wait}s: {e}")
                time.sleep(wait)
                
            except (self._APIError, self._APITimeoutError) as e:
                logger.warning(f"API error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
                    
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    raise
        
        return ""
    
    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """Batch generate (sequential)."""
        return [
            self.generate(p, system_prompt=system_prompt, **kwargs)
            for p in prompts
        ]


class VLLMOfflineLLM(BaseLLM):
    """
    Use vLLM directly in Python (offline mode).
    
    Better for single-machine, single-process usage.
    For multi-GPU or serving, use VLLMServerLLM instead.
    """
    
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        trust_remote_code: bool = True,
        quantization: Optional[str] = None,  # "awq", "gptq", "squeezellm"
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        
        from vllm import LLM, SamplingParams
        
        self.SamplingParams = SamplingParams
        
        logger.info(f"Loading vLLM model: {model_name}")
        logger.info(f"  tensor_parallel_size: {tensor_parallel_size}")
        logger.info(f"  gpu_memory_utilization: {gpu_memory_utilization}")
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            quantization=quantization,
            **kwargs
        )
        
        # Get tokenizer for chat template
        self.tokenizer = self.llm.get_tokenizer()
        
        logger.info("vLLM model loaded successfully")
    
    def _format_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        enable_thinking: bool = True  # Qwen3 thinking mode
    ) -> str:
        """Format prompt using chat template."""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Use chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                # Check if model supports enable_thinking (Qwen3)
                template_kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True
                }
                if "Qwen3" in self.model_name or "qwen3" in self.model_name.lower():
                    template_kwargs["enable_thinking"] = enable_thinking
                
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    **template_kwargs
                )
                return formatted
            except Exception as e:
                logger.debug(f"Chat template failed: {e}")
        
        # Fallback
        if system_prompt:
            return f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        return f"User: {prompt}\n\nAssistant:"
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        enable_thinking: bool = True,  # Qwen3 thinking mode
        **kwargs
    ) -> str:
        """Generate using vLLM offline mode."""
        
        formatted = self._format_prompt(prompt, system_prompt, enable_thinking)
        
        sampling_params = self.SamplingParams(
            temperature=temperature if temperature > 0 else 0.01,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        outputs = self.llm.generate([formatted], sampling_params)
        
        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text.strip()
        return ""
    
    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> List[str]:
        """Batch generate using vLLM (efficient batching)."""
        
        formatted_prompts = [
            self._format_prompt(p, system_prompt) 
            for p in prompts
        ]
        
        sampling_params = self.SamplingParams(
            temperature=temperature if temperature > 0 else 0.01,
            max_tokens=max_tokens,
            **kwargs
        )
        
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        
        results = []
        for output in outputs:
            if output.outputs:
                results.append(output.outputs[0].text.strip())
            else:
                results.append("")
        
        return results

