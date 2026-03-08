"""
Abstractive compressor wrapper.

Wraps a small instruction-following LLM (Qwen3-4B, Llama-3.2-3B, Gemma-3-4B)
as a prompt compressor by asking it to summarise the input into a target
number of tokens.  This matches the compression mechanism assumed by the
SoftCom attack in the COMA framework.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

SUMMARIZE_INSTRUCTION = (
    "Please help me to summarize the content above into {} tokens"
)

MODEL_NAME_MAP = {
    "qwen3-4b": "Qwen/Qwen3-4B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma-3-4b": "google/gemma-3-4b-it",
}


class AbstractiveCompressor:
    """
    Abstractive compressor using a small instruction-following LLM.

    The model is asked to summarise the input into a target token count
    computed from the compression rate.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        device: str = "cuda",
        max_new_tokens: int = 512,
    ):
        resolved = MODEL_NAME_MAP.get(model_name, model_name)
        self.model_name = resolved
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.model = AutoModelForCausalLM.from_pretrained(
            resolved, device_map="auto", torch_dtype=torch.float16,
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(resolved, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("AbstractiveCompressor loaded with model=%s", resolved)

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def compress(self, text: str, rate: float = 0.6) -> dict:
        """
        Compress text by asking the model to summarise it.

        Args:
            text: input text
            rate: fraction of tokens to retain (determines target_tokens)
        """
        original_tokens = self._count_tokens(text)
        target_tokens = max(10, int(original_tokens * rate))

        instruction = SUMMARIZE_INSTRUCTION.format(target_tokens)
        prompt = f"{text}\n\n{instruction}"

        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=min(self.max_new_tokens, target_tokens + 50),
                do_sample=False,
            )
        generated = output_ids[0][inputs["input_ids"].size(1):]
        compressed_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        compressed_tokens = self._count_tokens(compressed_text)

        return {
            "compressed_text": compressed_text,
            "compressed_prompt": compressed_text,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
        }

    def compress_prompt(self, text: str, rate: float = 0.6, **kwargs) -> dict:
        """Alias for compatibility with LLMLingua-style interface."""
        return self.compress(text, rate=rate)

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
