"""
GCG-style attacks targeting small LLMs (Qwen3-4B, Llama-3.2-3B, etc.).

The small LLM acts as a compressor via a summarisation instruction:
    "Please help me to summarize the content above into {N} tokens"
where N depends on the compression rate / compression budget.

Two modes:
  GCG-1  – AttackforSmallLM
           Direct perturbation: replace tokens in the input (system prompt)
           so that the model's summarisation produces the target output.

  GCG-2  – MultiplePromptsAttackforSmallLM
           Universal adversarial suffix: optimise a single suffix that,
           appended to ANY of the provided prompts, causes the model's
           summarisation to produce the corresponding target output.

Prompt structure (GCG-1):
    [prompt_tokens] [instruction_tokens] [target_tokens]
     ^--- control ---^                    ^--- loss ---^

Prompt structure (GCG-2):
    [prompt_tokens] [instruction_tokens] [suffix_tokens] [target_tokens]
                                          ^-- control --^  ^--- loss ---^

Both classes follow the same .step() interface so they plug directly
into run_gcg_attack_smalllm().
"""

import math
import logging
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .gcg_utils import (
    AttackConfig,
    get_embedding_matrix,
    get_token_embeddings,
    get_nonascii_toks_stable,
    sample_control,
)

log = logging.getLogger(__name__)

# Default summarisation instruction template
SUMMARIZE_INSTRUCTION = "Please help me to summarize the content above into {} tokens"


# ======================================================================
#  Shared loss helpers
# ======================================================================

def _target_output_loss(logits: torch.Tensor, target_slice: slice,
                        ids: torch.Tensor) -> torch.Tensor:
    """
    Standard next-token cross-entropy between model logits and target tokens.

    Because autoregressive logits at position t predict token t+1,
    the loss_slice is shifted by -1 from target_slice.
    """
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
    crit = nn.CrossEntropyLoss(reduction="none")
    loss = crit(
        logits[:, loss_slice, :].transpose(1, 2),
        ids[:, target_slice],
    )
    return loss.mean()


def _target_output_loss_per_example(logits: torch.Tensor, target_slice: slice,
                                    ids: torch.Tensor) -> torch.Tensor:
    """Per-example loss — returns [B] instead of scalar."""
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
    crit = nn.CrossEntropyLoss(reduction="none")
    loss = crit(
        logits[:, loss_slice, :].transpose(1, 2),
        ids[:, target_slice],
    )
    return loss.mean(dim=1)


# ======================================================================
#  GCG-1: Direct perturbation (single-prompt)
# ======================================================================

class AttackforSmallLM:
    """
    GCG-1: optimise token replacements **within the input** so that a
    small CausalLM's summarisation produces the desired target output.

    Prompt structure:
        [prompt_tokens] [instruction_tokens] [target_tokens]
        ^-- control --^  ^---- fixed -----^   ^--- loss ---^

    The instruction is: "Please help me to summarize the content above
    into {N} tokens" where N = compression_target_tokens.
    """

    def __init__(self, config: Optional[AttackConfig] = None,
                 compression_target_tokens: int = 200, **config_kwargs):
        if config is not None:
            self.config = config
        elif config_kwargs:
            self.config = AttackConfig(**config_kwargs)
        else:
            raise ValueError(
                "AttackforSmallLM requires config=AttackConfig(model_name='...') "
                "or model_name='...'"
            )

        self.compression_target_tokens = compression_target_tokens
        self.instruction_text = SUMMARIZE_INSTRUCTION.format(compression_target_tokens)

        self.model_name = self.config.model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", torch_dtype=torch.float16,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.nonascii_toks = get_nonascii_toks_stable(self.tokenizer)
        self.device = next(self.model.parameters()).device
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(self.config.seed)
        self.vocab_size = get_embedding_matrix(self.model).shape[0]

        # Pre-tokenise instruction (fixed across steps)
        self._instruction_ids = self.tokenizer.encode(
            self.instruction_text, add_special_tokens=False,
        )

        self.best_loss = float("inf")
        self.best_candidates = None

    # ---- slicing ----------------------------------------------------------

    def build_slices(self, prompt_len: int, instruction_len: int, target_len: int):
        """
        full_ids:  [prompt | instruction | target]
        control:   [0, prompt_len)                  — optimisable
        target:    [prompt_len + instruction_len, prompt_len + instruction_len + target_len)
        """
        control_slice = slice(0, prompt_len)
        target_slice = slice(
            prompt_len + instruction_len,
            prompt_len + instruction_len + target_len,
        )
        return control_slice, target_slice

    # ---- loss / gradient --------------------------------------------------

    def smalllm_loss(self, logits, control_slice, target_slice, ids):
        return _target_output_loss(logits, target_slice, ids)

    def smalllm_gradients(self, model, full_ids, control_slice, target_slice, device):
        """
        Gradient of loss w.r.t. one-hot token representations at control positions.
        """
        full_ids = full_ids.squeeze(0)  # [T]
        control_ids = full_ids[control_slice].to(device=device, dtype=torch.long)
        control_len = control_ids.numel()

        W = get_embedding_matrix(model)  # [V, D]
        base_embeds = get_token_embeddings(model, full_ids.unsqueeze(0)).detach()  # [1, T, D]

        one_hot = torch.zeros(
            (control_len, self.vocab_size), device=device, dtype=torch.float32,
        )
        one_hot.scatter_(1, control_ids.unsqueeze(1), 1.0)
        one_hot.requires_grad_(True)

        control_embeds = (one_hot.float() @ W.float()).to(dtype=W.dtype).unsqueeze(0)
        full_embeds = torch.cat([
            control_embeds,
            base_embeds[:, control_slice.stop:, :],
        ], dim=1)

        model.zero_grad(set_to_none=True)
        logits = model(inputs_embeds=full_embeds, input_ids=None).logits
        loss = self.smalllm_loss(logits, control_slice, target_slice, full_ids)
        loss.backward()

        return one_hot.grad.squeeze(0)  # [control_len, V]

    # ---- evaluation -------------------------------------------------------

    def evaluate_candidates(self, eval_logits, eval_full_ids, control_slice, target_slice):
        eval_batch_size = self.config.eval_batch_size
        batch_num = math.ceil(eval_full_ids.size(0) / eval_batch_size)
        losses = []
        with torch.no_grad():
            for i in range(batch_num):
                b_ids = eval_full_ids[i * eval_batch_size:(i + 1) * eval_batch_size]
                b_logits = eval_logits[i * eval_batch_size:(i + 1) * eval_batch_size]
                per_ex = _target_output_loss_per_example(b_logits, target_slice, b_ids)
                losses.append(per_ex.detach().cpu())
        losses = torch.cat(losses, dim=0).numpy()
        min_idx = int(np.argmin(losses))
        min_loss = float(losses[min_idx])
        return min_loss, min_idx

    # ---- single step ------------------------------------------------------

    def attack(self, prompt: str, target_output: str):
        """
        Run one GCG optimisation step.

        Args:
            prompt: the input text (system prompt) whose tokens will be perturbed.
            target_output: the desired summarisation the model should produce.

        Returns:
            (best_loss, best_control_ids_list)
        """
        device = self.device

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        instr_ids = self._instruction_ids
        target_ids = self.tokenizer.encode(target_output, add_special_tokens=False)

        control_slice, target_slice = self.build_slices(
            len(prompt_ids), len(instr_ids), len(target_ids),
        )

        full_ids = torch.tensor(
            prompt_ids + instr_ids + target_ids,
            dtype=torch.long, device=device,
        ).unsqueeze(0)  # [1, T]

        control_ids = full_ids[0, control_slice].clone()

        # 1) gradients
        grad = self.smalllm_gradients(
            self.model, full_ids, control_slice, target_slice, device,
        )
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        # 2) sample candidates
        candidates = sample_control(
            control_toks=control_ids,
            grad=grad,
            nonascii_toks=self.nonascii_toks,
            vocab_size=self.vocab_size,
            sample_batch_size=self.config.sample_batch_size,
            topk=self.config.top_k,
            generator=self.gen,
        )

        # 3) evaluate
        C = candidates.size(0)
        eval_full_ids = full_ids.expand(C, -1).clone()
        eval_full_ids[:, control_slice] = candidates
        eval_logits = self.model(input_ids=eval_full_ids).logits

        min_loss, min_idx = self.evaluate_candidates(
            eval_logits, eval_full_ids, control_slice, target_slice,
        )

        if min_loss < self.best_loss:
            self.best_loss = min_loss
            self.best_candidates = candidates[min_idx].clone()
        if self.best_candidates is None:
            self.best_candidates = candidates[0].clone()

        return self.best_loss, self.best_candidates.detach().cpu().numpy().tolist()

    def step(self, prompts: List[str], target_outputs: List[str]):
        """
        Uniform interface for run_gcg_attack_smalllm().
        GCG-1 is single-prompt, so unwrap the first element.
        """
        return self.attack(prompts[0], target_outputs[0])


# ======================================================================
#  GCG-2: Universal adversarial suffix (multi-prompt)
# ======================================================================

class MultiplePromptsAttackforSmallLM(AttackforSmallLM):
    """
    GCG-2: optimise a single adversarial suffix that, when appended to
    the summarisation instruction (after the system prompt which is not
    accessible), causes the model to produce the corresponding target output.

    Prompt structure per prompt i:
        [prompt_tokens_i] [instruction_tokens] [suffix_tokens] [target_tokens_i]
                                                ^-- control --^  ^--- loss ------^

    The suffix is placed right after the instruction. Gradients from all
    prompts are aggregated (mean) before candidate sampling.

    When the number of prompts is large, mini-batch gradient accumulation
    is used: prompts are split into groups of `prompt_batch_size`, gradients
    are computed per group, and then averaged.
    """

    def __init__(self, config: Optional[AttackConfig] = None,
                 compression_target_tokens: int = 200,
                 prompt_batch_size: Optional[int] = None,
                 **config_kwargs):
        super().__init__(config=config,
                         compression_target_tokens=compression_target_tokens,
                         **config_kwargs)
        self._suffix_init_text = "! " * self.config.suffix_length
        # prompt_batch_size: how many prompts per mini-batch for gradient /
        # evaluation.  None = process all prompts at once (original behaviour).
        self.prompt_batch_size = prompt_batch_size

    @staticmethod
    def build_slices_suffix(prompt_len: int, instruction_len: int,
                            suffix_len: int, target_len: int):
        """
        full_ids:  [prompt | instruction | suffix | target]
        control:   [prompt_len + instruction_len,
                    prompt_len + instruction_len + suffix_len)
        target:    [prompt_len + instruction_len + suffix_len, ...)
        """
        ctrl_start = prompt_len + instruction_len
        control_slice = slice(ctrl_start, ctrl_start + suffix_len)
        target_start = ctrl_start + suffix_len
        target_slice = slice(target_start, target_start + target_len)
        return control_slice, target_slice

    # ---- batch gradient ---------------------------------------------------

    def batch_smalllm_gradients(self, model, full_ids, suffix_slices,
                                target_slices, attention_mask, device):
        """
        Compute gradients w.r.t. shared suffix one-hot across a batch.
        Returns: grad [suffix_len, V], suffix_mask [B, T].
        """
        model.zero_grad(set_to_none=True)

        full_ids = full_ids.to(device=device, dtype=torch.long)
        attention_mask = attention_mask.to(device=device)
        B, T = full_ids.shape
        suffix_length = suffix_slices[0].stop - suffix_slices[0].start

        suffix_starts = torch.tensor(
            [sl.start for sl in suffix_slices], device=device, dtype=torch.long,
        )
        pos = suffix_starts[:, None] + torch.arange(suffix_length, device=device)

        base_embeds = get_token_embeddings(model, full_ids).detach()
        W = get_embedding_matrix(model)
        V, D = W.shape

        suffix_ids0 = full_ids[0, suffix_slices[0].start:suffix_slices[0].stop]
        one_hot = torch.zeros((suffix_length, V), device=device, dtype=torch.float32)
        one_hot.scatter_(1, suffix_ids0.unsqueeze(1), 1.0)
        one_hot.requires_grad_(True)

        suffix_embeds = (one_hot.float() @ W.float()).to(dtype=base_embeds.dtype)
        suffix_values = suffix_embeds.unsqueeze(0).expand(B, -1, -1)

        index = pos.unsqueeze(-1).expand(B, suffix_length, D)
        suffix_full = torch.zeros(B, T, D, device=device, dtype=base_embeds.dtype)
        suffix_full.scatter_(1, index, suffix_values)

        suffix_mask = torch.zeros_like(full_ids, dtype=torch.bool)
        suffix_mask.scatter_(1, pos, True)
        m = suffix_mask.unsqueeze(-1).to(dtype=base_embeds.dtype)

        full_embeds = base_embeds * (1.0 - m) + suffix_full

        logits = model(
            inputs_embeds=full_embeds, attention_mask=attention_mask,
        ).logits

        losses = []
        for b in range(B):
            per_loss = _target_output_loss(
                logits[b:b+1], target_slices[b], full_ids[b],
            )
            losses.append(per_loss)
        loss = torch.stack(losses).mean()
        loss.backward()

        return one_hot.grad, suffix_mask

    # ---- batch evaluation -------------------------------------------------

    def _eval_candidate_losses(self, model, eval_full_ids, eval_attention_mask,
                               target_slices, B):
        """
        Evaluate candidates and return per-candidate mean loss as CPU tensor [C].

        eval_full_ids: [C * B, T]   (C candidates × B prompts, interleaved)
        target_slices: list of B slices
        Returns: torch.Tensor of shape [C] on CPU.
        """
        C = eval_full_ids.size(0) // B
        eval_batch_size = self.config.eval_batch_size
        all_losses = []

        with torch.no_grad():
            for start in range(0, eval_full_ids.size(0), eval_batch_size):
                end = min(start + eval_batch_size, eval_full_ids.size(0))
                batch_ids = eval_full_ids[start:end]
                batch_mask = eval_attention_mask[start:end]
                logits = model(
                    input_ids=batch_ids, attention_mask=batch_mask,
                ).logits

                sub_losses = []
                for j in range(batch_ids.size(0)):
                    global_idx = start + j
                    b = global_idx % B
                    l = _target_output_loss_per_example(
                        logits[j:j+1], target_slices[b], batch_ids[j],
                    )
                    sub_losses.append(l)
                all_losses.append(torch.cat(sub_losses, dim=0))

        per_sample_losses = torch.cat(all_losses, dim=0).cpu()
        per_candidate_loss = per_sample_losses.reshape(C, B).mean(dim=1)
        return per_candidate_loss

    # ---- build eval ids ---------------------------------------------------

    def build_eval_full_ids(self, full_ids, candidates, suffix_slices,
                            attention_mask, device):
        from .hardcom_suffix import slice_to_pos

        full_ids = full_ids.to(device=device, dtype=torch.long)
        candidates = candidates.to(device=device, dtype=torch.long)
        B, T = full_ids.shape
        C, L = candidates.shape

        eval_full_ids = full_ids.unsqueeze(0).expand(C, -1, -1).clone().reshape(C * B, T)
        pos = slice_to_pos(suffix_slices, device)
        pos_repeat = pos.unsqueeze(0).expand(C, -1, -1).reshape(C * B, L)
        cand_repeat = candidates.unsqueeze(1).expand(C, B, L).reshape(C * B, L)
        rows = torch.arange(C * B, device=device).unsqueeze(1).expand(C * B, L)
        eval_full_ids[rows, pos_repeat] = cand_repeat
        eval_attention_mask = attention_mask.unsqueeze(0).expand(C, -1, -1).reshape(C * B, T)
        return eval_full_ids, eval_attention_mask

    # ---- attack -----------------------------------------------------------

    def attack(self, prompts: List[str], target_outputs: List[str]):
        """
        Run one GCG step: optimise shared suffix across all (prompt, target) pairs.
        The suffix is placed right after the instruction, before the target.

        When self.prompt_batch_size is set, prompts are processed in
        mini-batches for both gradient computation and candidate evaluation,
        enabling optimisation over large prompt sets without OOM.
        """
        device = self.device
        suffix_length = self.config.suffix_length
        instr_ids = self._instruction_ids

        # Initialise or reuse suffix
        suffix_init_ids = self.tokenizer.encode(
            self._suffix_init_text, add_special_tokens=False,
        )[:suffix_length]
        if len(suffix_init_ids) < suffix_length:
            pad_id = self.tokenizer.encode("!", add_special_tokens=False)[0]
            suffix_init_ids += [pad_id] * (suffix_length - len(suffix_init_ids))

        if self.best_candidates is not None:
            current_suffix_ids = self.best_candidates.cpu().tolist()
        else:
            current_suffix_ids = suffix_init_ids

        # Build per-prompt: [prompt | instruction | suffix | target]
        all_full_ids = []
        suffix_slices = []
        target_slices = []
        for prompt, target_output in zip(prompts, target_outputs):
            p_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            t_ids = self.tokenizer.encode(target_output, add_special_tokens=False)
            full = p_ids + instr_ids + current_suffix_ids + t_ids

            s_slice, t_slice = self.build_slices_suffix(
                len(p_ids), len(instr_ids), suffix_length, len(t_ids),
            )
            all_full_ids.append(full)
            suffix_slices.append(s_slice)
            target_slices.append(t_slice)

        # Pad to same length
        max_len = max(len(x) for x in all_full_ids)
        pad_id = self.tokenizer.pad_token_id or 0
        padded, masks = [], []
        for ids in all_full_ids:
            pad_len = max_len - len(ids)
            padded.append(ids + [pad_id] * pad_len)
            masks.append([1] * len(ids) + [0] * pad_len)

        full_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
        suffix_ids = full_ids[0, suffix_slices[0].start:suffix_slices[0].stop].clone()

        B = len(prompts)
        prompt_bs = self.prompt_batch_size or B  # default: all prompts at once

        # 1) Mini-batch gradient accumulation
        accumulated_grad = None
        num_mini = 0
        for mb_start in range(0, B, prompt_bs):
            mb_end = min(mb_start + prompt_bs, B)
            grad_mb, _ = self.batch_smalllm_gradients(
                self.model,
                full_ids[mb_start:mb_end],
                suffix_slices[mb_start:mb_end],
                target_slices[mb_start:mb_end],
                attention_mask[mb_start:mb_end],
                device,
            )
            if accumulated_grad is None:
                accumulated_grad = grad_mb.detach().clone()
            else:
                accumulated_grad += grad_mb.detach()
            num_mini += 1

        grad = accumulated_grad / num_mini
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        # 2) sample candidates
        candidates = sample_control(
            control_toks=suffix_ids,
            grad=grad,
            nonascii_toks=self.nonascii_toks,
            vocab_size=self.vocab_size,
            sample_batch_size=self.config.sample_batch_size,
            topk=self.config.top_k,
            generator=self.gen,
        )

        # 3) Mini-batch candidate evaluation
        C = candidates.size(0)
        accumulated_losses = torch.zeros(C)
        for mb_start in range(0, B, prompt_bs):
            mb_end = min(mb_start + prompt_bs, B)
            mb_B = mb_end - mb_start
            eval_ids, eval_mask = self.build_eval_full_ids(
                full_ids[mb_start:mb_end],
                candidates,
                suffix_slices[mb_start:mb_end],
                attention_mask[mb_start:mb_end],
                device,
            )
            mb_losses = self._eval_candidate_losses(
                self.model, eval_ids, eval_mask,
                target_slices[mb_start:mb_end], mb_B,
            )
            accumulated_losses += mb_losses * mb_B

        per_candidate_loss = accumulated_losses / B
        min_idx = int(per_candidate_loss.argmin().item())
        min_loss = float(per_candidate_loss[min_idx])

        if min_loss < self.best_loss:
            self.best_loss = min_loss
            self.best_candidates = candidates[min_idx].clone()
        if self.best_candidates is None:
            self.best_candidates = candidates[0].clone()

        return self.best_loss, self.best_candidates.detach().cpu().numpy().tolist()

    def step(self, prompts: List[str], target_outputs: List[str]):
        """Uniform interface for run_gcg_attack_smalllm()."""
        return self.attack(prompts, target_outputs)
