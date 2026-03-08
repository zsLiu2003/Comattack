"""
HardCom Stage II – context-editing attack for extractive compressors.

Given original context x_ctx and target x̃_tgt (= x_ctx with important words W
removed), find a perturbation of x_ctx so the surrogate compressor drops W.

Two surrogate types:
  - LLMLingua1 (PPL-based): minimise cross-entropy at W positions → W becomes
    predictable → low PPL → compressor drops W under budget.
  - LLMLingua2 (token classification): push classifier to predict "drop"
    (label=0) at W positions.

Attack levels:
  - token-level: edit individual tokens in x_ctx
  - word-level:  edit whole words in x_ctx
"""

import math
import copy
import logging
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification

from .gcg_utils import (
    AttackConfig,
    get_embedding_matrix,
    get_token_embeddings,
    get_nonascii_toks_stable,
    sample_control,
    NPEncoder,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Slice helpers
# ──────────────────────────────────────────────────────────────────────────

def find_span_token_slice(
    tokenizer,
    text: str,
    span_start: int,
    span_end: int,
) -> slice:
    """
    Map a character-level span [span_start, span_end) in *text* to a
    token-level slice using the tokenizer's offset mapping.
    """
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]  # list of (char_start, char_end) per token

    tok_start, tok_end = None, None
    for i, (cs, ce) in enumerate(offsets):
        if tok_start is None and ce > span_start:
            tok_start = i
        if cs < span_end:
            tok_end = i + 1

    if tok_start is None or tok_end is None:
        raise ValueError(
            f"Cannot map char span [{span_start}, {span_end}) to tokens. "
            f"Text length={len(text)}, num_tokens={len(offsets)}"
        )
    return slice(tok_start, tok_end)


def find_control_and_target_slices(
    tokenizer,
    context: str,
    removed_span: str,
    span_start: int,
    edit_radius: int = -1,
) -> Tuple[slice, slice]:
    """
    Compute token-level slices for the control (editable) and target (to-drop)
    regions.

    Args:
        tokenizer:    HF tokenizer
        context:      original context string
        removed_span: the answer / keyword text that was removed
        span_start:   character offset of removed_span in context
        edit_radius:  how many tokens around the target are editable.
                      -1 = entire context is editable.

    Returns:
        (control_slice, target_slice) — both are token-level slices
    """
    span_end = span_start + len(removed_span)
    target_slice = find_span_token_slice(tokenizer, context, span_start, span_end)

    if edit_radius < 0:
        # entire context is editable
        n_tokens = len(tokenizer(context, add_special_tokens=False)["input_ids"])
        control_slice = slice(0, n_tokens)
    else:
        ctrl_start = max(0, target_slice.start - edit_radius)
        n_tokens = len(tokenizer(context, add_special_tokens=False)["input_ids"])
        ctrl_end = min(n_tokens, target_slice.stop + edit_radius)
        control_slice = slice(ctrl_start, ctrl_end)

    return control_slice, target_slice


# ──────────────────────────────────────────────────────────────────────────
# LLMLingua1 context-edit attack
# ──────────────────────────────────────────────────────────────────────────

class ContextEditAttackLLMLingua1:
    """
    HardCom context-editing attack using an LLMLingua1 (PPL-based) surrogate.

    The surrogate scores each token by its cross-entropy (surprise).  Tokens
    with LOW cross-entropy (predictable) are dropped first.  To make the
    compressor drop target words W, we edit nearby tokens so that W becomes
    more predictable (lower CE) in the surrogate's scoring.

    Loss = CE at target positions (minimise → W predictable → dropped).
    Control = editable tokens in the context.
    """

    def __init__(self, config: Optional[AttackConfig] = None, **config_kwargs):
        if config is not None:
            self.config = config
        elif config_kwargs:
            self.config = AttackConfig(**config_kwargs)
        else:
            raise ValueError("Requires config or keyword arguments")

        self.model_name = self.config.model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.nonascii_toks = get_nonascii_toks_stable(self.tokenizer)
        self.device = self.model.device
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(self.config.seed)
        self.vocab_size = get_embedding_matrix(self.model).shape[0]
        self.best_loss = float("inf")
        self.best_control_ids = None

    # ── loss ──────────────────────────────────────────────────────────
    @staticmethod
    def _target_ce_loss(logits, target_slice, ids):
        """
        Cross-entropy at target positions (the words we want the compressor
        to DROP).  Minimising this makes them predictable → low PPL → dropped.
        """
        # autoregressive: logits[t-1] predicts ids[t]
        shift_logits = logits[:, target_slice.start - 1 : target_slice.stop - 1, :]
        shift_labels = ids[:, target_slice.start : target_slice.stop]
        if ids.dim() == 1:
            shift_labels = shift_labels.unsqueeze(0)
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="mean",
        )
        return loss

    # ── gradients ─────────────────────────────────────────────────────
    def compute_gradients(self, full_ids, control_slice, target_slice):
        """
        Gradient of target CE loss w.r.t. one-hot control representation.
        Returns tensor [control_len, vocab_size].
        """
        device = self.device
        full_ids = full_ids.squeeze(0).to(device)
        control_ids = full_ids[control_slice].clone()
        ctrl_len = control_ids.numel()

        W = get_embedding_matrix(self.model)
        base_embeds = get_token_embeddings(self.model, full_ids.unsqueeze(0)).detach()

        one_hot = torch.zeros(ctrl_len, self.vocab_size, device=device, dtype=torch.float32)
        one_hot.scatter_(1, control_ids.unsqueeze(1), 1.0)
        one_hot.requires_grad_(True)

        ctrl_embeds = (one_hot.float() @ W.float()).to(dtype=W.dtype).unsqueeze(0)
        full_embeds = torch.cat([
            base_embeds[:, :control_slice.start, :],
            ctrl_embeds,
            base_embeds[:, control_slice.stop:, :],
        ], dim=1)

        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs_embeds=full_embeds).logits
        loss = self._target_ce_loss(logits, target_slice, full_ids.unsqueeze(0))
        loss.backward()

        return one_hot.grad.detach(), loss.item()

    # ── single step ───────────────────────────────────────────────────
    def attack_step(self, full_ids, control_slice, target_slice):
        """
        One GCG step: compute gradients → sample candidates → evaluate → pick best.
        Returns (best_loss, best_control_ids_list).
        """
        device = self.device
        full_ids = full_ids.squeeze(0).to(device)

        grad, _ = self.compute_gradients(full_ids, control_slice, target_slice)
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        control_ids = full_ids[control_slice].clone()
        candidates = sample_control(
            control_toks=control_ids,
            grad=grad,
            nonascii_toks=self.nonascii_toks,
            vocab_size=self.vocab_size,
            sample_batch_size=self.config.sample_batch_size,
            topk=self.config.top_k,
            generator=self.gen,
        )  # [C, ctrl_len]

        # evaluate
        C = candidates.size(0)
        eval_ids = full_ids.unsqueeze(0).expand(C, -1).clone()
        eval_ids[:, control_slice] = candidates.to(device)

        losses = []
        bs = self.config.eval_batch_size
        with torch.no_grad():
            for i in range(0, C, bs):
                batch = eval_ids[i : i + bs]
                logits = self.model(input_ids=batch).logits
                for j in range(logits.size(0)):
                    l = self._target_ce_loss(
                        logits[j : j + 1], target_slice, batch[j : j + 1]
                    )
                    losses.append(l.item())
        losses = np.array(losses)

        min_idx = int(np.argmin(losses))
        min_loss = float(losses[min_idx])

        if min_loss < self.best_loss:
            self.best_loss = min_loss
            self.best_control_ids = candidates[min_idx].clone()
        if self.best_control_ids is None:
            self.best_control_ids = candidates[0].clone()

        return self.best_loss, self.best_control_ids.cpu().numpy().tolist()

    # ── full attack loop ──────────────────────────────────────────────
    def attack(
        self,
        context: str,
        removed_span: str,
        span_start: int,
        edit_radius: int = -1,
        num_steps: Optional[int] = None,
    ) -> dict:
        """
        Run the full context-editing attack.

        Args:
            context:      original context string
            removed_span: the answer / keyword text to make the compressor drop
            span_start:   character offset of removed_span in context
            edit_radius:  token radius around target that is editable (-1 = all)
            num_steps:    override config.num_steps

        Returns:
            dict with best_loss, attacked_context, control_ids, loss_history
        """
        control_slice, target_slice = find_control_and_target_slices(
            self.tokenizer, context, removed_span, span_start, edit_radius
        )

        full_ids = self.tokenizer(
            context, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)

        steps = num_steps or self.config.num_steps
        loss_history = []

        for step in range(1, steps + 1):
            loss, ctrl_ids = self.attack_step(full_ids, control_slice, target_slice)
            loss_history.append(loss)

            # update full_ids with best control tokens for next step
            full_ids[0, control_slice] = self.best_control_ids.to(self.device)

            if step % max(1, steps // 10) == 0:
                logger.info("step %d/%d  loss=%.4f  best=%.4f", step, steps, loss, self.best_loss)

        attacked_context = self.tokenizer.decode(
            full_ids[0], skip_special_tokens=True
        )

        return {
            "best_loss": self.best_loss,
            "attacked_context": attacked_context,
            "control_ids": ctrl_ids,
            "loss_history": loss_history,
            "control_slice": (control_slice.start, control_slice.stop),
            "target_slice": (target_slice.start, target_slice.stop),
        }


# ──────────────────────────────────────────────────────────────────────────
# LLMLingua2 context-edit attack
# ──────────────────────────────────────────────────────────────────────────

class ContextEditAttackLLMLingua2:
    """
    HardCom context-editing attack using an LLMLingua2 (token classification)
    surrogate.

    The surrogate predicts keep(1)/drop(0) per token.  To make it drop W,
    we edit nearby tokens so that the classifier predicts label=0 at W
    positions.

    Loss = token-classification CE with label=0 at target positions.
    Control = editable tokens in the context.
    """

    def __init__(self, config: Optional[AttackConfig] = None, **config_kwargs):
        if config is not None:
            self.config = config
        elif config_kwargs:
            self.config = AttackConfig(**config_kwargs)
        else:
            raise ValueError("Requires config or keyword arguments")

        self.model_name = self.config.model_name
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=True
        )
        self.nonascii_toks = get_nonascii_toks_stable(self.tokenizer)
        self.device = self.model.device
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(self.config.seed)
        self.vocab_size = get_embedding_matrix(self.model).shape[0]
        self.best_loss = float("inf")
        self.best_control_ids = None

    # ── loss ──────────────────────────────────────────────────────────
    @staticmethod
    def _target_cls_loss(logits, target_slice, ids):
        """
        Token classification loss: predict label=0 (drop) at target positions.
        No shift needed — TokenClassification directly maps tokens to labels.
        """
        target_logits = logits[:, target_slice.start : target_slice.stop, :]
        B, T, C = target_logits.shape
        labels = torch.zeros(B, T, device=target_logits.device, dtype=torch.long)
        loss = F.cross_entropy(
            target_logits.reshape(-1, C), labels.reshape(-1), reduction="mean"
        )
        return loss

    # ── gradients ─────────────────────────────────────────────────────
    def compute_gradients(self, full_ids, control_slice, target_slice):
        """
        Gradient of classification loss w.r.t. control embeddings,
        projected to vocab space via W^T.
        Returns tensor [control_len, vocab_size].
        """
        device = self.device
        full_ids = full_ids.squeeze(0).to(device)

        W = get_embedding_matrix(self.model)
        base_embeds = get_token_embeddings(self.model, full_ids.unsqueeze(0)).detach()

        ctrl_embeds = base_embeds[:, control_slice.start : control_slice.stop, :].clone().detach()
        ctrl_embeds.requires_grad_(True)

        full_embeds = torch.cat([
            base_embeds[:, :control_slice.start, :],
            ctrl_embeds,
            base_embeds[:, control_slice.stop:, :],
        ], dim=1)

        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs_embeds=full_embeds).logits
        loss = self._target_cls_loss(logits, target_slice, full_ids.unsqueeze(0))
        loss.backward()

        # project embedding-space gradient to vocab space
        grad = ctrl_embeds.grad.squeeze(0) @ W.float().t()
        return grad.detach(), loss.item()

    # ── single step ───────────────────────────────────────────────────
    def attack_step(self, full_ids, control_slice, target_slice):
        device = self.device
        full_ids = full_ids.squeeze(0).to(device)

        grad, _ = self.compute_gradients(full_ids, control_slice, target_slice)
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        control_ids = full_ids[control_slice].clone()
        candidates = sample_control(
            control_toks=control_ids,
            grad=grad,
            nonascii_toks=self.nonascii_toks,
            vocab_size=self.vocab_size,
            sample_batch_size=self.config.sample_batch_size,
            topk=self.config.top_k,
            generator=self.gen,
        )

        C = candidates.size(0)
        eval_ids = full_ids.unsqueeze(0).expand(C, -1).clone()
        eval_ids[:, control_slice] = candidates.to(device)

        losses = []
        bs = self.config.eval_batch_size
        with torch.no_grad():
            for i in range(0, C, bs):
                batch = eval_ids[i : i + bs]
                logits = self.model(input_ids=batch).logits
                for j in range(logits.size(0)):
                    l = self._target_cls_loss(
                        logits[j : j + 1], target_slice, batch[j : j + 1]
                    )
                    losses.append(l.item())
        losses = np.array(losses)

        min_idx = int(np.argmin(losses))
        min_loss = float(losses[min_idx])

        if min_loss < self.best_loss:
            self.best_loss = min_loss
            self.best_control_ids = candidates[min_idx].clone()
        if self.best_control_ids is None:
            self.best_control_ids = candidates[0].clone()

        return self.best_loss, self.best_control_ids.cpu().numpy().tolist()

    # ── full attack loop ──────────────────────────────────────────────
    def attack(
        self,
        context: str,
        removed_span: str,
        span_start: int,
        edit_radius: int = -1,
        num_steps: Optional[int] = None,
    ) -> dict:
        control_slice, target_slice = find_control_and_target_slices(
            self.tokenizer, context, removed_span, span_start, edit_radius
        )

        full_ids = self.tokenizer(
            context, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)

        steps = num_steps or self.config.num_steps
        loss_history = []

        for step in range(1, steps + 1):
            loss, ctrl_ids = self.attack_step(full_ids, control_slice, target_slice)
            loss_history.append(loss)
            full_ids[0, control_slice] = self.best_control_ids.to(self.device)

            if step % max(1, steps // 10) == 0:
                logger.info("step %d/%d  loss=%.4f  best=%.4f", step, steps, loss, self.best_loss)

        attacked_context = self.tokenizer.decode(
            full_ids[0], skip_special_tokens=True
        )
        return {
            "best_loss": self.best_loss,
            "attacked_context": attacked_context,
            "control_ids": ctrl_ids,
            "loss_history": loss_history,
            "control_slice": (control_slice.start, control_slice.stop),
            "target_slice": (target_slice.start, target_slice.stop),
        }


# ──────────────────────────────────────────────────────────────────────────
# Orchestration: run context-edit attack on a dataset
# ──────────────────────────────────────────────────────────────────────────

def run_context_edit_attack(
    attacker,  # ContextEditAttackLLMLingua1 or ContextEditAttackLLMLingua2
    dataset: list,
    task: str,
    edit_radius: int = -1,
    num_steps: Optional[int] = None,
    output_path: Optional[str] = None,
) -> list:
    """
    Run context-editing attack across a dataset.

    Args:
        attacker:    initialised attack object
        dataset:     list of dicts.  Required keys depend on task:
                     - "qa": context, target_context, removed_span, span_start
                     - "pref": original_context, target_context, deleted_words, ...
                     - "spc"/"guardrail": system_prompt, removed_phrases, ...
        task:        "qa", "pref", "spc", or "guardrail"
        edit_radius: token radius around target (-1 = all tokens editable)
        num_steps:   override config.num_steps
        output_path: if given, save incremental results as JSONL

    Returns:
        list of result dicts with attacked_context, best_loss, etc.
    """
    import json
    from tqdm import tqdm

    results = []
    fh = None
    if output_path:
        fh = open(output_path, "w", encoding="utf-8")

    for idx, entry in enumerate(tqdm(dataset, desc=f"Context-edit ({task})")):
        if task == "qa":
            context = entry["context"]
            removed_span = entry.get("removed_span", "")
            span_start = entry.get("span_start", -1)
            if not removed_span or span_start < 0:
                results.append({**entry, "attacked_context": context, "best_loss": None, "skip": True})
                continue
        elif task == "pref":
            context = entry.get("original_context", "")
            removed_span = " ".join(entry.get("deleted_words", []))
            # find span_start of the first deleted word in context
            span_start = context.find(removed_span) if removed_span else -1
            if not removed_span or span_start < 0:
                results.append({**entry, "attacked_context": context, "best_loss": None, "skip": True})
                continue
        elif task in ("spc", "guardrail"):
            context = entry.get("system_prompt", "")
            removed_phrases = entry.get("removed_phrases", [])
            if not removed_phrases:
                results.append({**entry, "attacked_context": context, "best_loss": None, "skip": True})
                continue
            # use the first removed phrase as the target span
            removed_span = removed_phrases[0]
            span_start = context.find(removed_span)
            if span_start < 0:
                results.append({**entry, "attacked_context": context, "best_loss": None, "skip": True})
                continue
        else:
            raise ValueError(f"Unknown task: {task}")

        # reset attacker state for each entry
        attacker.best_loss = float("inf")
        attacker.best_control_ids = None

        result = attacker.attack(
            context=context,
            removed_span=removed_span,
            span_start=span_start,
            edit_radius=edit_radius,
            num_steps=num_steps,
        )

        out = {**entry, **result}
        results.append(out)

        if fh:
            fh.write(json.dumps(out, cls=NPEncoder, ensure_ascii=False) + "\n")
            fh.flush()

    if fh:
        fh.close()
        logger.info("Saved %d results to %s", len(results), output_path)

    return results
