"""
Base LLM Interface
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class BaseLLM(ABC):
    """Abstract base class for all LLM providers."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        Generate response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """Generate responses for multiple prompts."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"




