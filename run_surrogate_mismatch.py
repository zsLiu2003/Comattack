"""
RQ3 -- Surrogate Mismatch Study (Table 5).

For each (target compressor, surrogate model) pair, this script:
  1. Samples 100 instances per task (seeded)
  2. Generates Stage I targets
  3. Runs Stage II preimage search using the specified surrogate
  4. Evaluates end-to-end ASR on the true target compressor + backend LLM
  5. Saves per-instance results and aggregate ASR

Usage (single config):
  python run_rq3_surrogate_mismatch.py \
    --target-compressor selective_context \
    --surrogate-model NousResearch/Llama-2-7b-hf \
    --task qa \
    --data-pref data/pref_manipulation_filtered.json \
    --data-qa data/squad_qa_filtered.json \
    --data-spc data/guardrail_dataset.json \
    --output results/rq3_mismatch/

Usage (aggregate results into Table 5):
  python run_rq3_surrogate_mismatch.py --aggregate --output results/rq3_mismatch/
"""

import os
import sys
import csv
import json
import glob
import random
import argparse
import logging
from typing import List, Dict
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
)
log = logging.getLogger("rq3_surrogate_mismatch")

# ---- experiment grid (Table 5) -----------------------------------------

EXTRACTIVE_PPL_TARGETS = ["selective_context", "llmlingua1"]
EXTRACTIVE_PPL_SURROGATES = [
    "gpt2",
    "NousResearch/Llama-2-7b-hf",
    "microsoft/phi-2",
]

EXTRACTIVE_CLS_TARGETS = ["llmlingua2"]
EXTRACTIVE_CLS_SURROGATES = [
    "bert-base-uncased",
    "microsoft/deberta-v3-large",
    "xlm-roberta-large",
]

ABSTRACTIVE_TARGETS = ["llama-3.2-3b", "qwen3-4b", "gemma-3-4b"]
ABSTRACTIVE_SURROGATES = {
    "llama-3.2-3b": ["llama-3.2-3b", "qwen3-4b", "gemma-3-4b"],
    "qwen3-4b":     ["llama-3.2-3b", "qwen3-4b", "gemma-3-4b"],
    "gemma-3-4b":   ["llama-3.2-3b", "qwen3-4b", "gemma-3-4b"],
}

ABSTRACTIVE_HF_NAMES = {
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "gemma-3-4b": "google/gemma-3-4b-it",
}

ALL_TASKS = ["prom", "deg", "qa", "spc"]
SAMPLE_SIZE = 100


# ---- data loading and target generation ---------------------------------

def _load_and_sample(path, n, seed):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(seed)
    if len(data) > n:
        data = rng.sample(data, n)
    return data


def generate_targets(dataset, task):
    if task in ("prom", "deg"):
        from run_pref_attack import compute_pref_target_offline
        return [{**e, **compute_pref_target_offline(e)} for e in dataset]
    if task == "qa":
        from run_qa_attack import compute_qa_target
        return [{**e, **compute_qa_target(e)} for e in dataset]
    if task == "spc":
        from run_guardrail_attack import compute_guardrail_target
        return [{**e, **compute_guardrail_target(e)} for e in dataset]
    raise ValueError(f"Unknown task: {task}")


# ---- attack dispatchers -------------------------------------------------

def run_attack_extractive_ppl(dataset, task, surrogate_model, args):
    """PPL-based surrogate (for SelectiveContext and LLMLingua1)."""
    from comattack.attacks.gcg_utils import AttackConfig
    from comattack.attacks.hardcom_context_edit import (
        ContextEditAttackLLMLingua1,
        run_context_edit_attack,
    )
    config = AttackConfig(
        model_name=surrogate_model,
        num_steps=args.num_steps,
        sample_batch_size=args.batch_size,
        top_k=args.topk,
        eval_batch_size=args.eval_batch_size,
        test_steps=args.test_steps,
        seed=args.seed,
    )
    attacker = ContextEditAttackLLMLingua1(config=config)
    atk_task = {"prom": "pref", "deg": "pref", "qa": "qa", "spc": "spc"}[task]
    return run_context_edit_attack(
        attacker=attacker, dataset=dataset, task=atk_task,
        edit_radius=args.edit_radius, num_steps=args.num_steps,
    )


def run_attack_extractive_cls(dataset, task, surrogate_model, args):
    """Token-classification surrogate (for LLMLingua2)."""
    from comattack.attacks.gcg_utils import AttackConfig
    from comattack.attacks.hardcom_context_edit import (
        ContextEditAttackLLMLingua2,
        run_context_edit_attack,
    )
    config = AttackConfig(
        model_name=surrogate_model,
        num_steps=args.num_steps,
        sample_batch_size=args.batch_size,
        top_k=args.topk,
        eval_batch_size=args.eval_batch_size,
        test_steps=args.test_steps,
        seed=args.seed,
    )
    attacker = ContextEditAttackLLMLingua2(config=config)
    atk_task = {"prom": "pref", "deg": "pref", "qa": "qa", "spc": "spc"}[task]
    return run_context_edit_attack(
        attacker=attacker, dataset=dataset, task=atk_task,
        edit_radius=args.edit_radius, num_steps=args.num_steps,
    )


def run_attack_abstractive(dataset, task, surrogate_model, args):
    """Abstractive surrogate (SoftCom GCG-1)."""
    from comattack.attacks.gcg_utils import AttackConfig
    from comattack.attacks.softcom import AttackforSmallLM
    from comattack.attacks.orchestration import run_gcg_attack_smalllm

    hf_name = ABSTRACTIVE_HF_NAMES.get(surrogate_model, surrogate_model)
    config = AttackConfig(
        model_name=hf_name,
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

    results = []
    for idx, entry in enumerate(dataset):
        prompt, target_output = _get_softcom_io(entry, task)
        if prompt is None:
            results.append({**entry, "skip": True})
            continue

        attacker.best_loss = float("inf")
        attacker.best_candidates = None

        r = run_gcg_attack_smalllm(
            attacker=attacker,
            prompts=[prompt],
            target_outputs=[target_output],
            num_steps=args.num_steps,
            test_steps=args.test_steps,
            success_threshold=args.success_threshold,
        )

        attacked = prompt
        if r["best_suffix_ids"]:
            attacked = attacker.tokenizer.decode(
                r["best_suffix_ids"], skip_special_tokens=True
            )
        # use task-appropriate field name for attacked text
        atk_field = "attacked_prompt" if task == "spc" else "attacked_context"
        results.append({
            **entry,
            atk_field: attacked,
            "best_loss": r["best_loss"],
            "converged": r["converged"],
        })
    return results


def _get_softcom_io(entry, task):
    if task in ("prom", "deg"):
        p, t = entry.get("original_context", ""), entry.get("target_context", "")
    elif task == "qa":
        p, t = entry.get("context", ""), entry.get("target_context", "")
    elif task == "spc":
        p, t = entry.get("system_prompt", ""), entry.get("target_prompt", "")
    else:
        return None, None
    return (p, t) if p and t else (None, None)


# ---- e2e evaluation -----------------------------------------------------

def get_attacked_texts(results, task):
    texts = []
    for r in results:
        t = r.get("attacked_context") or r.get("attacked_prompt")
        if t is None:
            fallback = {"prom": "original_context", "deg": "original_context",
                        "qa": "context", "spc": "system_prompt"}
            t = r.get(fallback.get(task, ""), "")
        texts.append(t)
    return texts


def run_single_config(args):
    task = args.task
    target = args.target_compressor
    surrogate = args.surrogate_model

    log.info("=== target=%s  surrogate=%s  task=%s ===", target, surrogate, task)

    data_path = _resolve_data_path(args, task)
    dataset = _load_and_sample(data_path, SAMPLE_SIZE, args.seed)
    log.info("Sampled %d entries from %s", len(dataset), data_path)

    dataset = generate_targets(dataset, task)

    if target in EXTRACTIVE_PPL_TARGETS:
        results = run_attack_extractive_ppl(dataset, task, surrogate, args)
    elif target in EXTRACTIVE_CLS_TARGETS:
        results = run_attack_extractive_cls(dataset, task, surrogate, args)
    elif target in ABSTRACTIVE_TARGETS:
        results = run_attack_abstractive(dataset, task, surrogate, args)
    else:
        raise ValueError(f"Unknown target compressor: {target}")

    from comattack.evaluation.e2e_eval import (
        make_compressor, make_backend_llm, evaluate_dataset,
    )
    compressor = make_compressor(target)
    backend = make_backend_llm(args.backend_llm, provider=args.llm_provider)
    attacked_texts = get_attacked_texts(results, task)

    eval_result = evaluate_dataset(
        dataset=dataset, attacked_texts=attacked_texts,
        compressor=compressor, backend_llm=backend,
        task=task, compression_rate=args.compression_rate,
    )
    asr = eval_result["asr"]
    log.info("ASR = %.4f  (%d / %d)", asr,
             eval_result["successes"], eval_result["total"])

    # save
    safe_target = target.replace("/", "_")
    safe_surr = surrogate.replace("/", "_")
    out_dir = os.path.join(args.output, safe_target, safe_surr, task)
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "target_compressor": target,
        "surrogate_model": surrogate,
        "task": task,
        "sample_size": len(dataset),
        "asr": asr,
        "successes": eval_result["successes"],
        "total": eval_result["total"],
        "compression_rate": args.compression_rate,
        "backend_llm": args.backend_llm,
        "num_steps": args.num_steps,
        "seed": args.seed,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as f:
        for r, ev in zip(results, eval_result["per_instance"]):
            row = {**r, "e2e_success": ev["success"],
                   "llm_output": ev["llm_output"]}
            row.pop("loss_history", None)
            f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")

    log.info("Saved to %s", out_dir)
    return summary


def _resolve_data_path(args, task):
    if task in ("prom", "deg"):
        return args.data_pref
    if task == "qa":
        return args.data_qa
    if task == "spc":
        return args.data_spc
    raise ValueError(task)


# ---- aggregate into Table 5 ---------------------------------------------

def aggregate_results(output_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(output_dir, "*/*/*/summary.json"))):
        with open(path) as f:
            rows.append(json.load(f))
    if not rows:
        log.warning("No summary.json found under %s", output_dir)
        return

    grid = defaultdict(dict)
    for r in rows:
        key = (r["target_compressor"], r["surrogate_model"])
        grid[key][r["task"]] = r["asr"]

    print(f"{'Target':<20s} {'Surrogate':<30s} "
          f"{'Prom':>6s} {'Deg':>6s} {'QA':>6s} {'SPC':>6s} {'Avg':>6s}")
    print("-" * 100)

    csv_rows = [["target", "surrogate", "prom", "deg", "qa", "spc", "avg"]]
    for (tgt, surr), tasks in sorted(grid.items()):
        vals = [tasks.get(t, None) for t in ALL_TASKS]
        valid = [v for v in vals if v is not None]
        avg = sum(valid) / max(len(valid), 1) if valid else 0
        cells = [f"{v:.2f}" if v is not None else "--" for v in vals]
        print(f"{tgt:<20s} {surr:<30s} " +
              "  ".join(f"{c:>6s}" for c in cells) + f"  {avg:6.2f}")
        csv_rows.append([tgt, surr] + cells + [f"{avg:.2f}"])

    csv_path = os.path.join(output_dir, "table5_surrogate_mismatch.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"\nCSV saved to {csv_path}")


# ---- CLI ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RQ3 Surrogate Mismatch Study")

    p.add_argument("--target-compressor")
    p.add_argument("--surrogate-model")
    p.add_argument("--task", choices=ALL_TASKS)

    p.add_argument("--data-pref", default="data/pref_manipulation_filtered.json")
    p.add_argument("--data-qa", default="data/squad_qa_filtered.json")
    p.add_argument("--data-spc", default="data/guardrail_dataset.json")

    p.add_argument("--backend-llm", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--llm-provider", default="auto",
                   choices=["auto", "server", "offline"])
    p.add_argument("--compression-rate", type=float, default=0.6)

    p.add_argument("--output", default="results/rq3_mismatch/")
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

    p.add_argument("--aggregate", action="store_true",
                   help="Only aggregate existing results (skip attacks)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.aggregate:
        aggregate_results(args.output)
        return
    if not args.target_compressor or not args.surrogate_model or not args.task:
        log.error("--target-compressor, --surrogate-model, --task required")
        sys.exit(1)
    run_single_config(args)


if __name__ == "__main__":
    main()
