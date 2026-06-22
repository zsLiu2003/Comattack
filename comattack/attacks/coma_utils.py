import json
import math
import logging
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification)
class NPEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NPEncoder, self).default(obj)


def get_embedding_layer(model):
    emb = model.get_input_embeddings()   # nn.Embedding
    if emb is None:
        raise ValueError(f"{type(model)} has no input embeddings")

    return emb


def get_embedding_matrix(model):
    emb = model.get_input_embeddings()   # nn.Embedding
    if emb is None or getattr(emb, "weight", None) is None:
        raise ValueError(f"{type(model)} has no input embedding matrix")
    return emb.weight


def get_token_embeddings(model, input_ids, dtype=None):
    emb = model.get_input_embeddings()          # == wte/embed_tokens/embed_in
    if emb is None:
        raise ValueError(f"{type(model)} has no input embeddings")

    x = emb(input_ids)                          # [bs, seq, hidden]

    # optional: cast to a specific dtype (e.g. fp16)
    if dtype is not None:
        x = x.to(dtype)

    return x


def get_nonascii_toks_stable(tokenizer, device="cpu"):
    """
    Return token IDs that should be excluded from adversarial suffix sampling:
    non-ASCII tokens, non-printable tokens, and special tokens (BOS/EOS/PAD/UNK).
    """
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    excluded_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            excluded_toks.append(i)

    if tokenizer.bos_token_id is not None:
        excluded_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        excluded_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        excluded_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        excluded_toks.append(tokenizer.unk_token_id)

    return torch.tensor(excluded_toks, device=device)

def nth_find(haystack: str, needle: str, n: int = 0) -> int:
    """Return the start index of the n-th occurrence of needle in haystack; -1 if not found. n=0 means first."""
    if n < 0:
        raise ValueError("n must be >= 0")
    start = 0
    for _ in range(n + 1):
        idx = haystack.find(needle, start)
        if idx == -1:
            return -1
        start = idx + len(needle)
    return idx

def keyword_char_span_in_fulltext(full_text: str, sentence: str, keyword: str,
                                 sentence_occ: int = 0, keyword_occ: int = 0):
    """
    Find the slice position of the keyword in the full text.
    First: find the sentence in the full text
    Second: find the keyword in the sentence
    Third: return the slice position of the keyword in the full text
    """

    sent_start = nth_find(full_text, sentence, sentence_occ)
    if sent_start == -1:
        raise ValueError("sentence not found in full_text (must match exactly)")
    kw_in_sent = nth_find(sentence, keyword, keyword_occ)
    if kw_in_sent == -1:
        raise ValueError("keyword not found in sentence")
    char_start = sent_start + kw_in_sent
    char_end = char_start + len(keyword)
    return char_start, char_end



def _ppl_control_loss(logits: torch.Tensor, suffix_slice: slice, target_slice: slice, ids: torch.Tensor) -> torch.Tensor:
    """
    PPL-based control loss: cross-entropy over the suffix region.
    Supports both 1D ids [T] and 2D ids [B, T].
    """
    target_slice = suffix_slice
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    L = nn.CrossEntropyLoss(reduction='none')
    loss = L(logits[:, target_slice.start-1:target_slice.stop-1, :].transpose(1, 2), ids[:, target_slice])
    return loss.mean()

def _token_cls_loss(logits: torch.Tensor, suffix_slice: slice, target_slice: slice, ids: torch.Tensor) -> torch.Tensor:
    """
    Token-classification loss for LLMLingua2: predict label=0 (remove) at target positions.
    No shift: TokenClassification models predict labels for each input token directly.
    Supports both 1D ids [T] and 2D ids [B, T].
    """
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    L = nn.CrossEntropyLoss(reduction='none')
    labels = torch.full_like(ids, fill_value=-100, dtype=torch.long)
    labels[:, target_slice] = 0
    loss = L(logits[:, target_slice, :].transpose(1, 2), labels[:, target_slice])
    return loss.mean()


def find_suffix_slice(prompt: str, suffix_length: int = 20):
    """
    This function is used to find the suffix slice in the prompt.
    """
    return slice(len(prompt) - suffix_length, len(prompt))

def find_target_slice(prompt: str, guardrail_sentence: str, guardrail_keyword: str):
    """
    This function is used to find the target slice in the prompt.
    """
    target_slice = keyword_char_span_in_fulltext(prompt, guardrail_sentence, guardrail_keyword)
    return slice(target_slice[0], target_slice[1])

def find_all_slices(prompt: str, guardrail_sentence: str, guardrail_keyword: str):
    """
    This function is used to find all the slices in the prompt.
    """

    suffix_slice = find_suffix_slice(prompt)
    target_slice = find_target_slice(prompt, guardrail_sentence, guardrail_keyword)
    return suffix_slice, target_slice

class AttackConfig():
    """
    A class to configure the attack. All parameters are configurable at init.
    """
    def __init__(self,
        model_name: str,
        suffix_length: int = 20,
        sample_batch_size: int = 256,  # batch size for sampling candidates
        num_steps: int = 200,  # number of optimization steps
        top_k: int = 64,  # top-k tokens per position for sampling
        test_steps: int = 5,  # steps between evaluations
        eval_batch_size: int = 128,  # batch size for evaluating candidates
        seed: int = 1234,
        loss_weight1: float = 1.0,  # for LLMLingua2: token_cls loss weight
        loss_weight2: float = 1.0,   # for LLMLingua2: surrogate PPL loss weight
        surrogate_model_name: Optional[str] = None,  # CausalLM for PPL evaluation (LLMLingua2)
        **kwargs,  # allow extra config keys for future extensibility
    ):
        self.model_name = model_name
        self.suffix_length = suffix_length
        self.sample_batch_size = sample_batch_size
        self.num_steps = num_steps
        self.top_k = top_k
        self.test_steps = test_steps
        self.eval_batch_size = eval_batch_size
        self.seed = seed
        self.loss_weight1 = loss_weight1
        self.loss_weight2 = loss_weight2
        self.surrogate_model_name = surrogate_model_name
        self._extra = kwargs


def sample_control(
    control_toks: torch.Tensor,     # [L] long
    grad: torch.Tensor,             # [L, V] float/half
    nonascii_toks: torch.Tensor,
    vocab_size: int,
    sample_batch_size: int,
    topk: int = 256,
    sample_temp: float = 1.0,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """
    Sample candidate replacement tokens for the adversarial suffix.

    For each candidate, pick one random position in the suffix and replace it with
    a token drawn from the top-k(-gradient) candidates at that position.
    Sampling pool size is L * topk; sample_batch_size candidates are drawn.

    Returns: new_control_toks of shape [sample_batch_size, L].
    """
    device = control_toks.device

    if control_toks.dim() != 1:
        control_toks = control_toks.squeeze(0)
    if grad.dim() != 2:
        grad = grad.squeeze(0)
    L = control_toks.size(0)

    # sanitize grad and exclude non-ASCII tokens
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0).float()
    grad[:, nonascii_toks.to(device)] = float("inf")  # -grad = -inf, never in topk

    # top-k per position: top_indices [L, topk]
    top_vals, top_indices = (-grad).topk(topk, dim=1)

    # sample from pool of L * topk candidates
    pool_size = L * topk
    num_samples = min(sample_batch_size, pool_size)
    pick = torch.randperm(pool_size, device=device, generator=generator)[:num_samples]
    new_token_pos = pick // topk          # position in suffix
    k_idx = pick % topk                   # index within top-k

    # choose token within top-k: uniform or temperature-weighted
    if sample_temp == 1.0:
        new_token_val = top_indices[new_token_pos, k_idx]
    elif sample_temp <= 0:
        new_token_val = top_indices[new_token_pos, 0]
    else:
        logits = top_vals[new_token_pos, :] / float(sample_temp)
        probs = torch.softmax(logits, dim=1)
        k_sample = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
        new_token_val = top_indices[new_token_pos, k_sample]

    original = control_toks.unsqueeze(0).expand(num_samples, -1).clone()  # [num_samples, L]
    rows = torch.arange(num_samples, device=device)
    original[rows, new_token_pos] = new_token_val
    original.clamp_(0, vocab_size - 1)

    return original


def find_start_token(pos, encoding, text) -> Optional[int]:
    """
    Find the token index corresponding to character position `pos`.
    Falls back to scanning rightward up to 64 chars if the exact position is between tokens.
    """
    t = encoding.char_to_token(pos)
    if t is not None:
        return t
    for p in range(pos + 1, min(len(text), pos + 64)):
        t = encoding.char_to_token(p)
        if t is not None:
            return t
    return None

def find_end_token(pos_exclusive, encoding, text) -> Optional[int]:
    """
    Find the token index (exclusive) corresponding to character position `pos_exclusive`.
    Uses pos_exclusive-1 (last included char), falls back to scanning leftward.
    """
    pos = pos_exclusive - 1
    t = encoding.char_to_token(pos)
    if t is not None:
        return t + 1
    for p in range(pos - 1, max(-1, pos - 64), -1):
        t = encoding.char_to_token(p)
        if t is not None:
            return t + 1
    return None

def find_suffix_from_token(tokenizer, prompt, suffix_length: int = 20):
    """
    Find the token-level suffix slice (last `suffix_length` tokens) in the prompt.
    """
    full_ids = tokenizer(prompt, return_tensors="pt").input_ids  # [1, T]
    T = full_ids.size(1)
    return T - suffix_length, T

def find_slices_from_token(tokenizer, prompt, guardrail_sentence, guardrail_keyword, suffix_length: int = 20):
    """
    Find token-level suffix and target slices from character-level positions.
    """
    suffix_slice, target_slice = find_all_slices(prompt, guardrail_sentence, guardrail_keyword)

    target_slice_start, target_slice_stop = target_slice.start, target_slice.stop
    suffix_a, suffix_b = find_suffix_from_token(tokenizer, prompt, suffix_length)
    encoding = tokenizer(prompt, return_offsets_mapping=True)
    target_a = find_start_token(target_slice_start, encoding, prompt)
    target_b = find_end_token(target_slice_stop, encoding, prompt)

    if suffix_a is None or suffix_b is None or target_a is None or target_b is None or suffix_a >= suffix_b or target_a >= target_b:
        raise RuntimeError("Cannot stably map character span to token span (possibly whitespace or special-token edge case).")
    suffix_slice = slice(suffix_a, suffix_b)
    target_slice = slice(target_a, target_b)
    return suffix_slice, target_slice
