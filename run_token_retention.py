"""
RQ3 -- Critical Token Retention Study.

Analyzes how COMA steers the compressor's information retention by
measuring whether critical tokens are dropped after compression.

For each (compressor, task, instance) we measure:
  - CTRR (Critical Token Retention Rate):  fraction of critical tokens that
    survive compression.
  - NCTRR (Non-Critical Token Retention Rate):  fraction of non-critical
    tokens that survive.
  - Selectivity = NCTRR - CTRR after attack (how selectively COMA targets
    critical content).

Two conditions are compared:
  - Benign:   compress the original (unattacked) input
  - Attacked: compress the COMA-attacked input

For extractive compressors the comparison is token-level.  For abstractive
compressors we fall back to n-gram overlap of the critical span in the
compressed text.

Usage:
  python run_rq3_token_retention.py \
    --attack-results results/qa_hardcom_results.jsonl \
    --compressor llmlingua1 \
    --task qa \
    --compression-rate 0.6 \
    --output results/rq3_retention/

  python run_rq3_token_retention.py \
    --attack-results results/pref_hardcom_results.jsonl \
    --compressor selective_context \
    --task prom \
    --output results/rq3_retention/

  # Aggregate all results
  python run_rq3_token_retention.py --aggregate --output results/rq3_retention/
"""

import os
import sys
import csv
import json
import glob
import re
import argparse
import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
)
log = logging.getLogger("rq3_token_retention")

ALL_TASKS = ["prom", "deg", "qa", "spc"]


# =========================================================================
#  Critical token identification
# =========================================================================

def identify_critical_tokens_qa(entry: dict) -> Tuple[str, str, int]:
    """
    QA task: critical tokens are the answer span.

    Returns (original_text, critical_span, span_start_char).
    """
    context = entry.get("context", "")
    removed = entry.get("removed_span", "")
    span_start = entry.get("span_start", -1)
    if not removed or span_start < 0:
        answers = entry.get("answers", {})
        if answers:
            removed = answers.get("text", [""])[0]
            span_start = answers.get("answer_start", [-1])[0]
    return context, removed, span_start


def identify_critical_tokens_pref(entry: dict) -> Tuple[str, str, int]:
    """
    Preference task: critical tokens are the deleted_words from Stage I
    (words whose removal flips the LLM's preference).
    """
    context = entry.get("original_context", "")
    deleted_words = entry.get("deleted_words", [])
    critical_span = " ".join(deleted_words)
    span_start = context.find(critical_span) if critical_span else -1
    return context, critical_span, span_start


def identify_critical_tokens_spc(entry: dict) -> Tuple[str, str, int]:
    """
    SPC task: critical tokens are the guardrail negation phrases
    (e.g., 'never', 'must not', 'do not').

    Since removed_phrases may be scattered across the prompt, we
    concatenate them for n-gram overlap measurement. span_start
    is set to the position of the first phrase found.
    """
    prompt = entry.get("system_prompt", "")
    removed = entry.get("removed_phrases", [])
    if isinstance(removed, str):
        removed = [removed]
    # filter to phrases actually present in the prompt
    present = [p for p in removed if p and prompt.find(p) >= 0]
    if not present:
        return prompt, "", -1
    critical_span = " ".join(present)
    span_start = prompt.find(present[0])
    return prompt, critical_span, span_start


CRITICAL_TOKEN_FINDERS = {
    "qa":   identify_critical_tokens_qa,
    "prom": identify_critical_tokens_pref,
    "deg":  identify_critical_tokens_pref,
    "spc":  identify_critical_tokens_spc,
}


# =========================================================================
#  Retention measurement  -- extractive (token-level)
# =========================================================================

def _tokenize_words(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for retention analysis."""
    return re.findall(r"\b\w+\b", text.lower())


def measure_token_retention_extractive(
    original_text: str,
    compressed_text: str,
    critical_span: str,
) -> Dict[str, float]:
    """
    Measure token-level retention for extractive compressors.

    Returns dict with ctrr, nctrr, n_critical, n_noncritical.
    """
    orig_tokens = _tokenize_words(original_text)
    comp_tokens = set(_tokenize_words(compressed_text))
    crit_tokens = _tokenize_words(critical_span)
    crit_set = set(crit_tokens)

    if not crit_set:
        return {"ctrr": float("nan"), "nctrr": float("nan"),
                "n_critical": 0, "n_noncritical": len(orig_tokens)}

    # count how many unique critical tokens appear in compressed output
    crit_retained = sum(1 for t in crit_set if t in comp_tokens)
    ctrr = crit_retained / len(crit_set)

    non_crit_set = set(t for t in orig_tokens if t not in crit_set)
    if non_crit_set:
        nc_retained = sum(1 for t in non_crit_set if t in comp_tokens)
        nctrr = nc_retained / len(non_crit_set)
    else:
        nctrr = float("nan")

    return {
        "ctrr": ctrr,
        "nctrr": nctrr,
        "n_critical": len(crit_set),
        "n_noncritical": len(non_crit_set),
    }


# =========================================================================
#  Retention measurement -- abstractive (n-gram overlap)
# =========================================================================

def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def measure_token_retention_abstractive(
    original_text: str,
    compressed_text: str,
    critical_span: str,
) -> Dict[str, float]:
    """
    Measure retention for abstractive compressors via n-gram overlap.

    CTRR = fraction of critical unigrams + bigrams present in compressed text.
    NCTRR = same for non-critical tokens.
    """
    comp_lower = compressed_text.lower()
    crit_tokens = _tokenize_words(critical_span)
    orig_tokens = _tokenize_words(original_text)

    if not crit_tokens:
        return {"ctrr": float("nan"), "nctrr": float("nan"),
                "n_critical": 0, "n_noncritical": len(orig_tokens)}

    crit_set = set(crit_tokens)

    # unigram + bigram overlap for critical tokens
    crit_uni = sum(1 for t in crit_tokens if t in comp_lower)
    crit_bi = _ngrams(crit_tokens, 2)
    crit_bi_hit = sum(1 for bg in crit_bi if " ".join(bg) in comp_lower)
    ctrr = (crit_uni + crit_bi_hit) / (len(crit_tokens) + max(len(crit_bi), 1))

    # non-critical
    non_crit = [t for t in orig_tokens if t not in crit_set]
    if non_crit:
        nc_uni = sum(1 for t in set(non_crit) if t in comp_lower)
        nctrr = nc_uni / len(set(non_crit))
    else:
        nctrr = float("nan")

    return {
        "ctrr": ctrr,
        "nctrr": nctrr,
        "n_critical": len(crit_tokens),
        "n_noncritical": len(set(non_crit)),
    }


# =========================================================================
#  Compressor helpers
# =========================================================================

EXTRACTIVE_COMPRESSORS = {
    "selective_context", "sc", "llmlingua1", "llmlingua", "llmlingua2",
}

def is_extractive(compressor_name: str) -> bool:
    return compressor_name in EXTRACTIVE_COMPRESSORS


def make_compressor(name: str):
    from comattack.evaluation.e2e_eval import make_compressor as _mc
    return _mc(name)


# =========================================================================
#  Main analysis loop
# =========================================================================

def run_retention_analysis(args):
    task = args.task
    compressor_name = args.compressor

    log.info("=== compressor=%s  task=%s ===", compressor_name, task)

    # load attack results
    entries = []
    with open(args.attack_results, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    log.info("Loaded %d entries from %s", len(entries), args.attack_results)

    if args.max_entries > 0:
        entries = entries[:args.max_entries]

    compressor = make_compressor(compressor_name)
    finder = CRITICAL_TOKEN_FINDERS.get(task)
    if finder is None:
        raise ValueError(f"Unknown task: {task}")

    measure_fn = (measure_token_retention_extractive
                  if is_extractive(compressor_name)
                  else measure_token_retention_abstractive)

    results = []
    for idx, entry in enumerate(entries):
        original_text, critical_span, span_start = finder(entry)
        if not critical_span:
            continue

        attacked_text = (entry.get("attacked_context")
                         or entry.get("attacked_prompt")
                         or original_text)

        # compress benign
        comp_benign = compressor.compress(original_text, rate=args.compression_rate)
        comp_benign_text = comp_benign.get(
            "compressed_prompt", comp_benign.get("compressed_text", ""))

        # compress attacked
        comp_attack = compressor.compress(attacked_text, rate=args.compression_rate)
        comp_attack_text = comp_attack.get(
            "compressed_prompt", comp_attack.get("compressed_text", ""))

        ret_benign = measure_fn(original_text, comp_benign_text, critical_span)
        ret_attack = measure_fn(attacked_text, comp_attack_text, critical_span)

        results.append({
            "idx": idx,
            "ctrr_benign": ret_benign["ctrr"],
            "nctrr_benign": ret_benign["nctrr"],
            "ctrr_attack": ret_attack["ctrr"],
            "nctrr_attack": ret_attack["nctrr"],
            "ctrr_drop": ret_benign["ctrr"] - ret_attack["ctrr"],
            "selectivity": ret_attack["nctrr"] - ret_attack["ctrr"],
            "n_critical": ret_benign["n_critical"],
            "n_noncritical": ret_benign["n_noncritical"],
            "critical_span": critical_span[:80],
        })

        if (idx + 1) % 50 == 0:
            log.info("Processed %d / %d", idx + 1, len(entries))

    # aggregate
    valid = [r for r in results
             if not (np.isnan(r["ctrr_benign"]) or np.isnan(r["ctrr_attack"]))]
    if not valid:
        log.warning("No valid results to aggregate")
        return

    ctrr_b = np.mean([r["ctrr_benign"] for r in valid])
    ctrr_a = np.mean([r["ctrr_attack"] for r in valid])
    nctrr_b = np.nanmean([r["nctrr_benign"] for r in valid])
    nctrr_a = np.nanmean([r["nctrr_attack"] for r in valid])
    drop = ctrr_b - ctrr_a
    selectivity = nctrr_a - ctrr_a

    log.info("CTRR  benign=%.3f  attack=%.3f  drop=%.3f", ctrr_b, ctrr_a, drop)
    log.info("NCTRR benign=%.3f  attack=%.3f", nctrr_b, nctrr_a)
    log.info("Selectivity (NCTRR_atk - CTRR_atk) = %.3f", selectivity)

    # save
    out_dir = os.path.join(args.output, compressor_name, task)
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "compressor": compressor_name,
        "task": task,
        "n_instances": len(valid),
        "compression_rate": args.compression_rate,
        "ctrr_benign": round(ctrr_b, 4),
        "ctrr_attack": round(ctrr_a, 4),
        "ctrr_drop": round(drop, 4),
        "nctrr_benign": round(nctrr_b, 4),
        "nctrr_attack": round(nctrr_a, 4),
        "selectivity": round(selectivity, 4),
    }
    with open(os.path.join(out_dir, "retention_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "retention_per_instance.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    log.info("Saved to %s", out_dir)
    return summary


# =========================================================================
#  Aggregate all retention results
# =========================================================================

def aggregate_retention(output_dir):
    rows = []
    for path in sorted(
        glob.glob(os.path.join(output_dir, "*/*/retention_summary.json"))
    ):
        with open(path) as f:
            rows.append(json.load(f))

    if not rows:
        log.warning("No retention_summary.json found under %s", output_dir)
        return

    print(f"{'Compressor':<20s} {'Task':<6s} "
          f"{'CTRR_b':>7s} {'CTRR_a':>7s} {'Drop':>7s} "
          f"{'NCTRR_b':>8s} {'NCTRR_a':>8s} {'Select':>7s}")
    print("-" * 80)

    csv_rows = [["compressor", "task", "ctrr_benign", "ctrr_attack",
                 "ctrr_drop", "nctrr_benign", "nctrr_attack", "selectivity"]]

    for r in sorted(rows, key=lambda x: (x["compressor"], x["task"])):
        print(f"{r['compressor']:<20s} {r['task']:<6s} "
              f"{r['ctrr_benign']:7.3f} {r['ctrr_attack']:7.3f} "
              f"{r['ctrr_drop']:7.3f} "
              f"{r['nctrr_benign']:8.3f} {r['nctrr_attack']:8.3f} "
              f"{r['selectivity']:7.3f}")
        csv_rows.append([
            r["compressor"], r["task"],
            f"{r['ctrr_benign']:.4f}", f"{r['ctrr_attack']:.4f}",
            f"{r['ctrr_drop']:.4f}",
            f"{r['nctrr_benign']:.4f}", f"{r['nctrr_attack']:.4f}",
            f"{r['selectivity']:.4f}",
        ])

    csv_path = os.path.join(output_dir, "token_retention_table.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"\nCSV saved to {csv_path}")


# =========================================================================
#  CLI
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="RQ3 Critical Token Retention Study"
    )

    p.add_argument("--attack-results",
                   help="Path to attack results JSONL (from run_*_attack.py)")
    p.add_argument("--compressor",
                   help="Compressor name used to compress the input")
    p.add_argument("--task", choices=ALL_TASKS)
    p.add_argument("--compression-rate", type=float, default=0.6)
    p.add_argument("--max-entries", type=int, default=-1)
    p.add_argument("--output", default="results/rq3_retention/")

    p.add_argument("--aggregate", action="store_true",
                   help="Only aggregate existing retention results")
    return p.parse_args()


def main():
    args = parse_args()
    if args.aggregate:
        aggregate_retention(args.output)
        return
    if not args.attack_results or not args.compressor or not args.task:
        log.error("--attack-results, --compressor, --task required")
        sys.exit(1)
    run_retention_analysis(args)


if __name__ == "__main__":
    main()
