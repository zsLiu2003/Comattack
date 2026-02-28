"""
GCG attack orchestration: run the full optimisation loop.

Two orchestrators:
  - run_gcg_attack()          : HardCom attacks (LLMLingua1/2) with ExternalEvaluator
  - run_gcg_attack_smalllm()  : SoftCom attacks (small LMs) with loss-based convergence
"""
import json
import os
import logging
from typing import Optional, List, Dict, Any

from tqdm import tqdm

from .external_evaluator import ExternalEvaluator, EvalResult
from .gcg_utils import find_slices_from_token

log = logging.getLogger(__name__)


# ── HardCom orchestrator ─────────────────────────────────────────────────

def run_gcg_attack(
    attacker,
    evaluator: ExternalEvaluator,
    prompts: List[str],
    guardrail_sentences: List[str],
    guardrail_keywords: List[str],
    num_steps: Optional[int] = None,
    test_steps: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full GCG optimisation loop with periodic external evaluation.

    Works with ANY attacker that exposes:
        - attacker.step(prompts, sentences, keywords) -> (loss, suffix_ids)
        - attacker.config  (AttackConfig with num_steps, test_steps, suffix_length)
        - attacker.tokenizer
        - attacker.best_loss, attacker.best_candidates

    Returns:
        dict with keys:
            best_loss, best_suffix_ids, converged, final_eval, history, step
    """
    config = attacker.config
    num_steps = num_steps or config.num_steps
    test_steps = test_steps or config.test_steps

    # Pre-compute suffix slices (needed by evaluator to inject suffix)
    suffix_slices = []
    for prompt, gs, gk in zip(prompts, guardrail_sentences, guardrail_keywords):
        ss, _ = find_slices_from_token(
            attacker.tokenizer, prompt, gs, gk, config.suffix_length
        )
        suffix_slices.append(ss)

    converged = False
    final_eval = None

    pbar = tqdm(range(1, num_steps + 1), desc="GCG", disable=not verbose)
    for step in pbar:
        # ---- one GCG optimisation step (uniform interface) ----
        loss, suffix_ids = attacker.step(
            prompts, guardrail_sentences, guardrail_keywords,
        )

        pbar.set_postfix(loss=f"{loss:.4f}", best=f"{attacker.best_loss:.4f}")

        # ---- periodic external evaluation ----
        if step % test_steps == 0 or step == num_steps:
            result = evaluator.evaluate(
                prompts=prompts,
                guardrail_sentences=guardrail_sentences,
                guardrail_keywords=guardrail_keywords,
                suffix_ids=suffix_ids,
                suffix_slices=suffix_slices,
                step=step,
            )
            final_eval = result

            if verbose:
                tqdm.write(
                    f"  [eval step {step}] success_rate={result.success_rate:.2%} "
                    f" all_removed={result.success}  loss={loss:.4f}"
                )

            if result.success:
                converged = True
                if verbose:
                    tqdm.write(
                        f"  >>> Attack converged at step {step} – "
                        f"all targeted slices removed by compression."
                    )
                break

    return {
        "best_loss": attacker.best_loss,
        "best_suffix_ids": attacker.best_candidates.detach().cpu().numpy().tolist()
                           if attacker.best_candidates is not None else suffix_ids,
        "converged": converged,
        "final_eval": final_eval,
        "history": evaluator.history,
        "step": step,
    }


# ── SoftCom orchestrator ─────────────────────────────────────────────────

def run_gcg_attack_smalllm(
    attacker,
    prompts: List[str],
    target_outputs: List[str],
    num_steps: Optional[int] = None,
    test_steps: Optional[int] = None,
    success_threshold: float = 0.5,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full GCG optimisation loop for small-LM attacks.

    Works with:
        - AttackforSmallLM                   (GCG-1, single prompt)
        - MultiplePromptsAttackforSmallLM    (GCG-2, universal suffix)

    The attacker internally appends the summarisation instruction
    ("Please help me to summarize the content above into N tokens")
    between the prompt/suffix and the target output.

    Args:
        attacker: attack object with .step(prompts, target_outputs) and .config
        prompts: list of input prompts (system prompts to compress)
        target_outputs: list of desired summarisation outputs
        num_steps: override config.num_steps
        test_steps: how often to log detailed info
        success_threshold: loss below this = success
        output_dir: if set, save incremental results to a JSONL file here
        verbose: show progress bar

    Returns:
        dict with keys:
            best_loss, best_suffix_ids, converged, step, loss_history
    """
    config = attacker.config
    num_steps = num_steps or config.num_steps
    test_steps = test_steps or config.test_steps

    loss_history = []
    converged = False

    # Set up incremental results file
    results_file = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, "incremental_results.jsonl")
        meta = {
            "type": "run_start",
            "model": config.model_name,
            "num_prompts": len(prompts),
            "num_steps": num_steps,
            "suffix_length": config.suffix_length,
            "sample_batch_size": config.sample_batch_size,
            "top_k": config.top_k,
            "compression_target_tokens": attacker.compression_target_tokens,
            "instruction": attacker.instruction_text,
            "prompts": prompts,
            "target_outputs": target_outputs,
        }
        with open(results_file, "w") as f:
            f.write(json.dumps(meta) + "\n")

    pbar = tqdm(range(1, num_steps + 1), desc="GCG-SmallLM", disable=not verbose)
    for step in pbar:
        # one GCG step
        loss, suffix_ids = attacker.step(prompts, target_outputs)
        loss_history.append(loss)

        pbar.set_postfix(loss=f"{loss:.4f}", best=f"{attacker.best_loss:.4f}")

        # periodic logging
        if step % test_steps == 0 or step == num_steps:
            suffix_text = attacker.tokenizer.decode(suffix_ids, skip_special_tokens=True)
            msg = (
                f"  [step {step}] loss={loss:.4f}  "
                f"best={attacker.best_loss:.4f}  "
                f"suffix/control={suffix_text[:60]}..."
            )
            if verbose:
                tqdm.write(msg)

            # save checkpoint
            if results_file:
                entry = {
                    "type": "checkpoint",
                    "step": step,
                    "loss": loss,
                    "best_loss": attacker.best_loss,
                    "suffix_ids": suffix_ids if isinstance(suffix_ids, list) else suffix_ids,
                    "suffix_text": suffix_text,
                }
                with open(results_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")

            if attacker.best_loss < success_threshold:
                converged = True
                if verbose:
                    tqdm.write(
                        f"  >>> Converged at step {step} – "
                        f"loss {attacker.best_loss:.4f} < threshold {success_threshold}"
                    )
                break

    result = {
        "best_loss": attacker.best_loss,
        "best_suffix_ids": (
            attacker.best_candidates.detach().cpu().numpy().tolist()
            if attacker.best_candidates is not None else suffix_ids
        ),
        "converged": converged,
        "step": step,
        "loss_history": loss_history,
    }

    # save final results
    if output_dir:
        final_path = os.path.join(output_dir, "final_result.json")
        final_result = {
            **result,
            "best_suffix_text": attacker.tokenizer.decode(
                result["best_suffix_ids"], skip_special_tokens=True,
            ),
            "prompts": prompts,
            "target_outputs": target_outputs,
            "model": config.model_name,
            "compression_target_tokens": attacker.compression_target_tokens,
        }
        with open(final_path, "w") as f:
            json.dump(final_result, f, indent=2)
        log.info("Final results saved to %s", final_path)

    return result
