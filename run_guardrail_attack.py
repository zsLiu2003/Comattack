"""
End-to-end runner for the System Prompt Corruption attack (Task 3).

Pipeline:
  1. Load guardrail dataset (system prompts with identified guardrail sentences)
  2. For each entry, compute x_tgt: remove negation from guardrail sentences
  3. Run Stage II preimage search to find x_atk
  4. Save results (attacked system prompts + metadata)

Supports both HardCom (extractive compressors) and SoftCom (abstractive).

BASE_MODEL="NousResearch/Llama-2-7b-hf"
Usage:
  python run_guardrail_attack.py \
    --data /path/to/guardrail_dataset.json \
    --compressor llmlingua1 \
    --surrogate-model NousResearch/Llama-2-7b-hf \
    --num-steps 200 \
    --output results/guardrail_llmlingua1/
"""

import os
import json
import argparse
import logging

from comattack.targets.guardrail import generate_system_prompt_target

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s",
)
log = logging.getLogger("run_guardrail_attack")


# -- target generation --------------------------------------------------------

def compute_guardrail_target(entry: dict) -> dict:
    """
    Remove negation from guardrail sentences in the system prompt.
    Returns dict with target_prompt, removed_phrases, modified_guardrails.
    """
    system_prompt = entry["system_prompt"]
    guardrail_list = entry["guardrail_list"]
    return generate_system_prompt_target(system_prompt, guardrail_list)


# -- HardCom attack (extractive compressors) ----------------------------------

def run_hardcom_guardrail(args, dataset):
    from comattack.attacks.gcg_utils import AttackConfig
    from comattack.attacks.hardcom_context_edit import (
        ContextEditAttackLLMLingua1,
        ContextEditAttackLLMLingua2,
        run_context_edit_attack,
    )

    config = AttackConfig(
        model_name=args.surrogate_model,
        num_steps=args.num_steps,
        sample_batch_size=args.batch_size,
        top_k=args.topk,
        eval_batch_size=args.eval_batch_size,
        test_steps=args.test_steps,
        seed=args.seed,
    )

    # augment dataset with target info
    aug_dataset = []
    for entry in dataset:
        target_info = compute_guardrail_target(entry)
        aug = {**entry, **target_info}
        if not target_info["removed_phrases"]:
            log.warning("No negation found in entry — skipping")
        aug_dataset.append(aug)

    if args.compressor == "llmlingua1":
        attacker = ContextEditAttackLLMLingua1(config=config)
    elif args.compressor == "llmlingua2":
        attacker = ContextEditAttackLLMLingua2(config=config)
    else:
        raise ValueError(f"Unknown compressor for HardCom: {args.compressor}")

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "guardrail_hardcom_results.jsonl")

    results = run_context_edit_attack(
        attacker=attacker,
        dataset=aug_dataset,
        task="spc",
        edit_radius=args.edit_radius,
        num_steps=args.num_steps,
        output_path=output_path,
    )

    n_success = sum(1 for r in results if not r.get("skip"))
    summary = {
        "task": "guardrail",
        "method": "hardcom",
        "compressor": args.compressor,
        "surrogate_model": args.surrogate_model,
        "num_entries": len(dataset),
        "num_attacked": n_success,
        "num_steps": args.num_steps,
    }
    summary_path = os.path.join(args.output, "guardrail_hardcom_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("HardCom guardrail attack finished. %d entries processed.", n_success)
    return results


# -- SoftCom attack (abstractive compressors / small LLMs) --------------------

def run_softcom_guardrail(args, dataset):
    from comattack.attacks.gcg_utils import AttackConfig
    from comattack.attacks.softcom import AttackforSmallLM
    from comattack.attacks.orchestration import run_gcg_attack_smalllm

    config = AttackConfig(
        model_name=args.surrogate_model,
        suffix_length=args.suffix_length,
        num_steps=args.num_steps,
        sample_batch_size=args.batch_size,
        top_k=args.topk,
        eval_batch_size=args.eval_batch_size,
        test_steps=args.test_steps,
        seed=args.seed,
    )

    attacker = AttackforSmallLM(
        config=config,
        compression_target_tokens=args.compression_target_tokens,
    )

    os.makedirs(args.output, exist_ok=True)
    all_results = []

    for idx, entry in enumerate(dataset):
        target_info = compute_guardrail_target(entry)
        if not target_info["removed_phrases"]:
            log.warning("Entry %d: no negation found — skipping", idx)
            all_results.append({**entry, **target_info, "skip": True})
            continue

        system_prompt = entry["system_prompt"]
        target_prompt = target_info["target_prompt"]

        attacker.best_loss = float("inf")
        attacker.best_candidates = None

        log.info("Entry %d/%d: removed=%s", idx + 1, len(dataset),
                 target_info["removed_phrases"][:3])

        result = run_gcg_attack_smalllm(
            attacker=attacker,
            prompts=[system_prompt],
            target_outputs=[target_prompt],
            num_steps=args.num_steps,
            test_steps=args.test_steps,
            success_threshold=args.success_threshold,
            output_dir=None,
            verbose=True,
        )

        if result["best_suffix_ids"]:
            attacked_prompt = attacker.tokenizer.decode(
                result["best_suffix_ids"], skip_special_tokens=True
            )
        else:
            attacked_prompt = system_prompt

        out = {
            **entry,
            **target_info,
            "attacked_prompt": attacked_prompt,
            "best_loss": result["best_loss"],
            "converged": result["converged"],
            "steps": result["step"],
        }
        all_results.append(out)

        results_path = os.path.join(args.output, "guardrail_softcom_results.jsonl")
        with open(results_path, "a") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    n_attacked = sum(1 for r in all_results if not r.get("skip"))
    n_converged = sum(1 for r in all_results if r.get("converged"))
    summary = {
        "task": "guardrail",
        "method": "softcom",
        "compressor": args.compressor,
        "surrogate_model": args.surrogate_model,
        "num_entries": len(dataset),
        "num_attacked": n_attacked,
        "num_converged": n_converged,
        "num_steps": args.num_steps,
        "compression_target_tokens": args.compression_target_tokens,
    }
    summary_path = os.path.join(args.output, "guardrail_softcom_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("SoftCom guardrail attack done. %d/%d converged.", n_converged, n_attacked)
    return all_results


# -- main ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="COMA System Prompt Corruption attack runner (Task 3)"
    )

    p.add_argument("--data", required=True,
                   help="Path to guardrail dataset JSON")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--max-entries", type=int, default=-1,
                   help="Limit entries (-1 = all)")

    p.add_argument("--compressor", required=True,
                   choices=["llmlingua1", "llmlingua2", "selective_context",
                            "qwen3-4b", "llama-3.2-3b", "gemma-3-4b"])
    p.add_argument("--surrogate-model", required=True,
                   help="HuggingFace model name for the surrogate")

    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--test-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--edit-radius", type=int, default=-1)
    p.add_argument("--suffix-length", type=int, default=20)
    p.add_argument("--compression-target-tokens", type=int, default=200)
    p.add_argument("--success-threshold", type=float, default=0.5)

    return p.parse_args()


def main():
    args = parse_args()

    log.info("Loading dataset from %s", args.data)
    with open(args.data, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.max_entries > 0:
        dataset = dataset[:args.max_entries]
    log.info("Loaded %d entries", len(dataset))

    extractive = {"llmlingua1", "llmlingua2", "selective_context"}
    abstractive = {"qwen3-4b", "llama-3.2-3b", "gemma-3-4b"}

    if args.compressor in extractive:
        run_hardcom_guardrail(args, dataset)
    elif args.compressor in abstractive:
        run_softcom_guardrail(args, dataset)
    else:
        raise ValueError(f"Unknown compressor: {args.compressor}")


if __name__ == "__main__":
    main()
