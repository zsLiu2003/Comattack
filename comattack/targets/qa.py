"""
Stage I target selection for QA (Task 2).

Removes the ground-truth answer span from the context passage to produce
the proxy target x_tgt. The compressor should ideally reproduce this
answer-free version after compression.
"""

import json
import re
import copy
from typing import List, Dict, Optional


def _remove_answer_from_context(
    context: str,
    answer_texts: List[str],
    answer_starts: List[int],
) -> dict:
    """
    Remove the ground-truth answer span from the context using answer_start
    to locate the exact position.

    Returns a dict with:
        target_context:  context with the answer span excised
        removed_span:    the exact substring that was removed
        span_start:      character offset where removal happened
    """
    start = answer_starts[0]
    answer = answer_texts[0]
    end = start + len(answer)

    if context[start:end] == answer:
        target_context = context[:start] + context[end:]
    else:
        idx = context.find(answer)
        if idx != -1:
            target_context = context[:idx] + context[idx + len(answer):]
            start = idx
        else:
            return {
                "target_context": context,
                "removed_span": "",
                "span_start": -1,
            }

    target_context = re.sub(r"\s{2,}", " ", target_context).strip()

    return {
        "target_context": target_context,
        "removed_span": answer,
        "span_start": start,
    }


def generate_qa_targets(
    dataset: List[Dict], output_path: Optional[str] = None
) -> List[Dict]:
    """
    For every QA entry, remove the ground-truth answer text from the context.

    Args:
        dataset:     list of QA dicts with keys {context, question, answers, ...}
        output_path: if provided, write the result JSON here

    Returns:
        list of dicts, each augmented with target_context, removed_span, span_start
    """
    results = []
    for entry in dataset:
        ctx = entry["context"]
        ans_texts = entry["answers"]["text"]
        ans_starts = entry["answers"]["answer_start"]

        info = _remove_answer_from_context(ctx, ans_texts, ans_starts)

        out = copy.deepcopy(entry)
        out["target_context"] = info["target_context"]
        out["removed_span"] = info["removed_span"]
        out["span_start"] = info["span_start"]
        results.append(out)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[QA] Saved {len(results)} target entries to {output_path}")

    return results
