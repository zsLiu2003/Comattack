"""
Selective Context compressor wrapper.

Selective Context (Li et al., 2023) ranks tokens/spans by self-information
(negative log-probability under a causal LM) and retains the most informative
content under the budget.  Lower self-information tokens are dropped first.

This wrapper exposes the same interface as the other compressor wrappers used
in the COMA evaluation pipeline.
"""

import logging
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class SelectiveContextCompressor:
    """
    Selective Context compressor using self-information scoring.

    Tokens with low self-information (highly predictable) are dropped first
    to meet the compression budget.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto"
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("SelectiveContext loaded with model=%s", model_name)

    def _compute_self_information(self, text: str) -> tuple:
        """Compute per-token self-information (negative log-prob)."""
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.model.device)
        n_tokens = input_ids.size(1)

        if n_tokens == 0:
            return [], []

        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits  # [1, T, V]

        log_probs = torch.log_softmax(logits, dim=-1)  # [1, T, V]

        self_info = []
        tokens = input_ids[0].tolist()
        # first token has no preceding context -> assign high self-info
        self_info.append(float("inf"))
        for t in range(1, n_tokens):
            token_id = tokens[t]
            si = -log_probs[0, t - 1, token_id].item()
            self_info.append(si)

        return tokens, self_info

    def compress(self, text: str, rate: float = 0.6) -> dict:
        """
        Compress text by retaining `rate` fraction of tokens
        (those with the highest self-information).
        """
        tokens, self_info = self._compute_self_information(text)
        n_tokens = len(tokens)

        if n_tokens == 0:
            return {
                "compressed_text": text,
                "compressed_prompt": text,
                "original_tokens": 0,
                "compressed_tokens": 0,
            }

        n_keep = max(1, int(n_tokens * rate))
        # sort by self-information descending; keep top-n_keep
        indices = np.argsort(self_info)[::-1][:n_keep]
        # restore original order
        indices = sorted(indices)
        kept_ids = [tokens[i] for i in indices]
        compressed_text = self.tokenizer.decode(kept_ids, skip_special_tokens=True)

        return {
            "compressed_text": compressed_text,
            "compressed_prompt": compressed_text,
            "original_tokens": n_tokens,
            "compressed_tokens": len(kept_ids),
            "kept_indices": indices,
        }

    def compress_prompt(self, text: str, rate: float = 0.6, **kwargs) -> dict:
        """Alias for compatibility with LLMLingua interface."""
        return self.compress(text, rate=rate)

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
