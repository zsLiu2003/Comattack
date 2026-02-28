"""
End-to-end runner for the QA attack (Task ②).

Pipeline:
  1. Load filtered QA dataset (from Comattack)
  2. For each entry, compute x̃_tgt: remove ground-truth answer from context
  3. Run Stage II preimage search to find x_atk
  4. Save results (attacked contexts + metadata)

Supports both HardCom (extractive compressors) and SoftCom (abstractive).

Usage:
  python run_qa_attack.py \
    --data /path/to/squad_qa_filtered_llmlingua1_max900.json \
    --compressor llmlingua1 \
    --surrogate-model NousResearch/Llama-2-7b-hf \
    --num-steps 200 \
    --output results/qa_llmlingua1/
"""

import os
import sys
import json
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s",
)
log = logging.getLogger("run_qa_attack")


# ── target generation ────────────────────────────────────────────────────

def compute_qa_target(entry: dict) -> dict:
    """
    Remove the ground-truth answer span from the context.
    Uses answer_start for precise span removal.
    Returns dict with target_context, removed_span, span_start.
    """
    import re
    context = entry["context"]
    ans_texts = entry["answers"]["text"]
    ans_starts = entry["answers"]["answer_start"]

    start = ans_starts[0]
    answer = ans_texts[0]
    end = start + len(answer)

    if context[start:end] == answer:
        target_context = context[:start] + context[end:]
    else:
        idx = context.find(answer)
        if idx != -1:
            target_context = context[:idx] + context[idx + len(answer):]
            start = idx
        else:
            return {"target_context": context, "removed_span": "", "span_start": -1}

    target_context = re.sub(r"\s{2,}", " ", target_context).strip()
    return {"target_context": target_context, "removed_span": answer, "span_start": start}


# ── HardCom attack (extractive compressors) ──────────────────────────────

def run_hardcom_qa(args, dataset):
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
        target_info = compute_qa_target(entry)
        aug = {**entry, **target_info}
        if not target_info["removed_span"]:
            log.warning("No answer span found for entry %s — skipping", entry.get("id"))
        aug_dataset.append(aug)

    if args.compressor == "llmlingua1":
        attacker = ContextEditAttackLLMLingua1(config=config)
    elif args.compressor == "llmlingua2":
        attacker = ContextEditAttackLLMLingua2(config=config)
    else:
        raise ValueError(f"Unknown compressor for HardCom: {args.compressor}")

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "qa_hardcom_results.jsonl")

    results = run_context_edit_attack(
        attacker=attacker,
        dataset=aug_dataset,
        task="qa",
        edit_radius=args.edit_radius,
        num_steps=args.num_steps,
        output_path=output_path,
    )

    # save summary
    n_success = sum(1 for r in results if r.get("best_loss") is not None and not r.get("skip"))
    summary = {
        "task": "qa",
        "method": "hardcom",
        "compressor": args.compressor,
        "surrogate_model": args.surrogate_model,
        "num_entries": len(dataset),
        "num_attacked": n_success,
        "num_steps": args.num_steps,
    }
    summary_path = os.path.join(args.output, "qa_hardcom_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("HardCom QA attack finished. %d entries processed.", n_success)
    return results


# ── SoftCom attack (abstractive compressors / small LLMs) ────────────────

def run_softcom_qa(args, dataset):
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
        target_info = compute_qa_target(entry)
        if not target_info["removed_span"]:
            log.warning("No answer span for entry %s — skipping", entry.get("id"))
            all_results.append({**entry, "skip": True})
            continue

        context = entry["context"]
        target_context = target_info["target_context"]

        # reset attacker state
        attacker.best_loss = float("inf")
        attacker.best_candidates = None

        log.info("Entry %d/%d: answer=%r", idx + 1, len(dataset), target_info["removed_span"])

        result = run_gcg_attack_smalllm(
            attacker=attacker,
            prompts=[context],
            target_outputs=[target_context],
            num_steps=args.num_steps,
            test_steps=args.test_steps,
            success_threshold=args.success_threshold,
            output_dir=None,
            verbose=True,
        )

        # decode attacked context
        if result["best_suffix_ids"]:
            attacked_context = attacker.tokenizer.decode(
                result["best_suffix_ids"], skip_special_tokens=True
            )
        else:
            attacked_context = context

        out = {
            **entry,
            **target_info,
            "attacked_context": attacked_context,
            "best_loss": result["best_loss"],
            "converged": result["converged"],
            "steps": result["step"],
        }
        all_results.append(out)

        # incremental save
        results_path = os.path.join(args.output, "qa_softcom_results.jsonl")
        with open(results_path, "a") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # save summary
    n_attacked = sum(1 for r in all_results if not r.get("skip"))
    n_converged = sum(1 for r in all_results if r.get("converged"))
    summary = {
        "task": "qa",
        "method": "softcom",
        "compressor": args.compressor,
        "surrogate_model": args.surrogate_model,
        "num_entries": len(dataset),
        "num_attacked": n_attacked,
        "num_converged": n_converged,
        "num_steps": args.num_steps,
        "compression_target_tokens": args.compression_target_tokens,
    }
    summary_path = os.path.join(args.output, "qa_softcom_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("SoftCom QA attack finished. %d/%d entries converged.", n_converged, n_attacked)
    return all_results


# ── main ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="COMA QA attack runner (Task ②)")

    # data
    p.add_argument("--data", required=True, help="Path to filtered QA dataset JSON")
    p.add_argument("--output", required=True, help="Output directory for results")
    p.add_argument("--max-entries", type=int, default=-1, help="Limit entries (-1 = all)")

    # compressor / method
    p.add_argument("--compressor", required=True,
                   choices=["llmlingua1", "llmlingua2", "selective_context",
                            "qwen3-4b", "llama-3.2-3b", "gemma-3-4b"],
                   help="Target compressor")
    p.add_argument("--surrogate-model", required=True,
                   help="HuggingFace model name for the surrogate")

    # attack hyperparameters
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256, help="Candidate sample batch size")
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--test-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--edit-radius", type=int, default=-1,
                   help="Token radius around target for HardCom (-1 = all)")
    p.add_argument("--suffix-length", type=int, default=20,
                   help="Suffix length for SoftCom / GCG-2")
    p.add_argument("--compression-target-tokens", type=int, default=200,
                   help="Target token count for SoftCom compression instruction")
    p.add_argument("--success-threshold", type=float, default=0.5,
                   help="Loss threshold for SoftCom convergence")

    return p.parse_args()


def main():
    args = parse_args()

    log.info("Loading dataset from %s", args.data)
    with open(args.data, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.max_entries > 0:
        dataset = dataset[:args.max_entries]
    log.info("Loaded %d entries", len(dataset))

    # dispatch based on compressor type
    extractive = {"llmlingua1", "llmlingua2", "selective_context"}
    abstractive = {"qwen3-4b", "llama-3.2-3b", "gemma-3-4b"}

    if args.compressor in extractive:
        run_hardcom_qa(args, dataset)
    elif args.compressor in abstractive:
        run_softcom_qa(args, dataset)
    else:
        raise ValueError(f"Unknown compressor: {args.compressor}")


if __name__ == "__main__":
    main()
