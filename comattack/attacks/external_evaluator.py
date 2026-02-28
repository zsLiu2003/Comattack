import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field


# ======================================================================
#  EvalResult + ExternalEvaluator
# ======================================================================

@dataclass
class EvalResult:
    """Result of one external evaluation round."""
    step: int
    success: bool
    success_rate: float
    per_prompt: List[bool]
    compressed_texts: List[str]
    extra: Dict[str, Any] = field(default_factory=dict)


def _find_nth_eval(text: str, keyword: str, n: int = 0) -> int:
    """Return start index of the n-th (0-based) occurrence of keyword in text."""
    start = 0
    for _ in range(n + 1):
        idx = text.find(keyword, start)
        if idx == -1:
            return -1
        start = idx + len(keyword)
    return idx


def _extract_context_words(sentence, keyword, keyword_occ=0, n_words=2):
    """Return (left_words, right_words) around the keyword_occ-th occurrence."""
    lower_sent = sentence.lower()
    kw_start = _find_nth_eval(lower_sent, keyword.lower(), keyword_occ)
    if kw_start == -1:
        return [], []
    kw_end = kw_start + len(keyword)
    left_text = sentence[:kw_start].strip()
    right_text = sentence[kw_end:].strip()
    left_words = left_text.split()[-n_words:] if left_text else []
    right_words = right_text.split()[:n_words] if right_text else []
    return left_words, right_words


def _keyword_removed_in_context(compressed_text, keyword, guardrail_sentence, keyword_occ=0, n_context=2):
    """Check whether the specific keyword occurrence was removed, using context bigrams."""
    c_lower = compressed_text.lower()
    kw_lower = keyword.lower()
    if kw_lower not in c_lower:
        return True
    left_words, right_words = _extract_context_words(guardrail_sentence, keyword, keyword_occ, n_context)
    patterns = []
    for lw in left_words:
        patterns.append(f"{lw} {keyword}".lower())
    for rw in right_words:
        patterns.append(f"{keyword} {rw}".lower())
    if not patterns:
        return kw_lower not in c_lower
    return not any(p in c_lower for p in patterns)


class ExternalEvaluator:
    """Compress prompt with adversarial suffix, check if target keyword was removed."""

    def __init__(self, tokenizer, method, compress_fn, suffix_start_marker="",
                 control_init="!!!!!!!!!!!!!!!!", compression_target_token=None,
                 compression_rate=None, n_context=2):
        self.tokenizer = tokenizer
        self.method = method
        self.compress_fn = compress_fn
        self.suffix_start_marker = suffix_start_marker
        self.control_init = control_init
        self.compression_target_token = compression_target_token
        self.compression_rate = compression_rate
        self.n_context = n_context
        self.log = logging.getLogger(self.__class__.__name__)
        self.history: List[EvalResult] = []

    def _insert_suffix_into_prompt(self, prompt, suffix_ids, suffix_slice):
        token_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        token_ids[suffix_slice.start:suffix_slice.stop] = suffix_ids
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def evaluate(self, prompts, guardrail_sentences, guardrail_keywords,
                 suffix_ids, suffix_slices, step=-1, keyword_occs=None):
        N = len(prompts)
        assert N == len(guardrail_sentences) == len(guardrail_keywords) == len(suffix_slices)
        if keyword_occs is None:
            keyword_occs = [0] * N
        per_prompt, compressed_texts = [], []
        for i, (prompt, sentence, keyword, sl, kw_occ) in enumerate(
            zip(prompts, guardrail_sentences, guardrail_keywords, suffix_slices, keyword_occs)
        ):
            attacked_prompt = self._insert_suffix_into_prompt(prompt, suffix_ids, sl)
            if self.method == "llmlingua1":
                compressed = self.compress_fn(attacked_prompt, target_token=self.compression_target_token, instruction="", question="")
            elif self.method == "llmlingua2":
                compressed = self.compress_fn(attacked_prompt, rate=self.compression_rate)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            compressed_text = compressed["compressed_prompt"]
            compressed_texts.append(compressed_text)
            removed = _keyword_removed_in_context(compressed_text, keyword, sentence, kw_occ, self.n_context)
            per_prompt.append(removed)
            self.log.debug("step=%d prompt=%d keyword=%r occ=%d removed=%s", step, i, keyword, kw_occ, removed)
        success_rate = sum(per_prompt) / max(len(per_prompt), 1)
        result = EvalResult(step=step, success=all(per_prompt), success_rate=success_rate,
                            per_prompt=per_prompt, compressed_texts=compressed_texts)
        self.history.append(result)
        self.log.info("step=%d  success_rate=%.2f  all_removed=%s", step, success_rate, result.success)
        return result
