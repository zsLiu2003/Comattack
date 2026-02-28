"""
AutoCompressor Provider - For soft compression behavioral evaluation
====================================================================

Uses the native AutoCompressor model with soft embeddings for evaluation.

Usage:
    llm = AutoCompressorLLM(model_path="princeton-nlp/AutoCompressor-Llama-2-7b-6k")
    
    # Load soft embeddings for a specific prompt
    soft_embed = soft_embeddings[prompt_idx]  # Shape: [50, 4096]
    
    # Generate with soft prompt
    response = llm.generate_with_soft_prompt(
        user_query="...",
        soft_embedding=soft_embed
    )
"""

import logging
import torch
from typing import Optional, List, Union
from pathlib import Path

from .base import BaseLLM

logger = logging.getLogger(__name__)


class AutoCompressorLLM(BaseLLM):
    """
    AutoCompressor model for soft prompt evaluation.
    
    This model can:
    1. Generate responses using soft embeddings as system prompt
    2. Generate responses using regular text prompt (for comparison)
    """
    
    def __init__(
        self,
        model_name: str = "princeton-nlp/AutoCompressor-Llama-2-7b-6k",
        device: str = None,
        max_new_tokens: int = 512,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        
        # Lazy load
        self.model = None
        self.tokenizer = None
        
    def _load_model(self):
        """Load AutoCompressor model and tokenizer."""
        if self.model is not None:
            return
        
        logger.info(f"Loading AutoCompressor model: {self.model_name}")
        
        try:
            from auto_compressor import LlamaAutoCompressorModel, AutoCompressorTokenizer
            
            self.model = LlamaAutoCompressorModel.from_pretrained(
                self.model_name,
                dtype=torch.bfloat16,
                device_map=self.device
            )
            self.tokenizer = AutoCompressorTokenizer.from_pretrained(self.model_name)
            
            logger.info("AutoCompressor model loaded successfully")
            
        except ImportError:
            logger.error("auto_compressor not installed. Install from: https://github.com/princeton-nlp/AutoCompressor")
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        Generate using regular text prompt (no soft embeddings).
        For comparison with soft prompt generation.
        """
        self._load_model()
        
        # Build input
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = f"User: {prompt}\nAssistant:"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.01,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def generate_with_soft_prompt(
        self,
        user_query: str,
        soft_embedding: torch.Tensor,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """
        Generate using soft embeddings as system prompt.
        
        Args:
            user_query: The user's query text
            soft_embedding: Soft prompt embeddings [num_tokens, hidden_size]
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        self._load_model()
        
        # Ensure soft embedding is on correct device
        if soft_embedding.device != self.device:
            soft_embedding = soft_embedding.to(self.device)
        
        # Add batch dimension if needed
        if soft_embedding.dim() == 2:
            soft_embedding = soft_embedding.unsqueeze(0)  # [1, num_tokens, hidden_size]
        
        # Tokenize user query
        user_text = f"\n\nUser: {user_query}\nAssistant:"
        user_inputs = self.tokenizer(user_text, return_tensors="pt").to(self.device)
        
        # Get user embeddings
        user_embeds = self.model.get_input_embeddings()(user_inputs.input_ids)
        
        # Concatenate soft prompt + user embeddings
        # [1, soft_len + user_len, hidden_size]
        combined_embeds = torch.cat([soft_embedding, user_embeds], dim=1)
        
        # Create attention mask
        soft_len = soft_embedding.shape[1]
        user_len = user_embeds.shape[1]
        attention_mask = torch.ones(1, soft_len + user_len, device=self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.01,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Try to extract just the assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response
    
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
    
    def cleanup(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("AutoCompressor model cleaned up")




