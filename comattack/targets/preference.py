"""
Stage I target selection for Preference Manipulation (Task 1).

Greedy word deletion with a binary LLM judge: iteratively remove words
from the currently-preferred candidate until the judge's preference flips
to the attacker's target candidate.
"""

import json
import copy
from typing import List, Dict, Callable, Optional


def _concat_demos(entry: Dict) -> str:
    """Concatenate demo_1 ... demo_5 into a single context string."""
    parts = []
    for i in range(1, 6):
        key = f"demo_{i}"
        if key in entry:
            parts.append(entry[key])
    return "\n\n".join(parts)


def greedy_word_deletion(
    text: str,
    judge_fn: Callable[[str], str],
    target_label: str,
    max_deletions: int = -1,
) -> dict:
    """
    Stage I greedy word deletion for preference manipulation.

    Starting from `text`, iteratively delete one word at a time.
    After each deletion, call `judge_fn(current_text)` which returns the
    label of the preferred candidate (e.g. "demo_1").
    Stop as soon as `judge_fn` returns `target_label`.

    Args:
        text:          the candidate descriptions (compressed or original)
        judge_fn:      callable(text) -> predicted preference label (e.g. "demo_3")
        target_label:  the attacker's desired preference label
        max_deletions: max words to try deleting (-1 = no limit)

    Returns:
        dict with target_context, deleted_words, n_deletions, success
    """
    words = text.split()
    deleted_words = []
    success = False

    limit = len(words) if max_deletions < 0 else min(max_deletions, len(words))

    for _ in range(limit):
        best_idx = None
        for i in range(len(words)):
            trial = words[:i] + words[i + 1:]
            trial_text = " ".join(trial)
            prediction = judge_fn(trial_text)
            if prediction == target_label:
                best_idx = i
                break

        if best_idx is not None:
            deleted_words.append(words[best_idx])
            words = words[:best_idx] + words[best_idx + 1:]
            success = True
            break
        else:
            deleted_words.append(words[-1])
            words = words[:-1]

    target_context = " ".join(words)
    return {
        "target_context": target_context,
        "deleted_words": deleted_words,
        "n_deletions": len(deleted_words),
        "success": success,
    }


def generate_pref_targets(
    dataset: List[Dict],
    judge_fn: Callable[[str, str], str],
    compress_fn: Optional[Callable[[str], str]] = None,
    max_deletions: int = 50,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """
    Generate x_tgt for preference manipulation via greedy deletion.

    Args:
        dataset:      list of pref-manipulation dicts with demo_1..demo_5,
                      best, target, category
        judge_fn:     callable(context_text, question_text) -> predicted label
        compress_fn:  optional compressor; if given, first compress the demos
        max_deletions: max number of words to delete per entry
        output_path:  if provided, write the result JSON here

    Returns:
        list of dicts augmented with target_context, deleted_words, etc.
    """
    results = []
    for idx, entry in enumerate(dataset):
        if "best" not in entry or "target" not in entry:
            out = copy.deepcopy(entry)
            out["target_context"] = None
            out["_skip_reason"] = "missing best/target fields"
            results.append(out)
            continue

        context = _concat_demos(entry)
        question = entry.get("question", "")

        if compress_fn is not None:
            working_text = compress_fn(context)
        else:
            working_text = context

        target_label = entry["target"]

        def _judge(text, q=question):
            return judge_fn(text, q)

        info = greedy_word_deletion(
            text=working_text,
            judge_fn=_judge,
            target_label=target_label,
            max_deletions=max_deletions,
        )

        out = copy.deepcopy(entry)
        out["original_context"] = context
        out["target_context"] = info["target_context"]
        out["deleted_words"] = info["deleted_words"]
        out["n_deletions"] = info["n_deletions"]
        out["target_found"] = info["success"]
        results.append(out)

        if (idx + 1) % 10 == 0:
            print(f"[Pref] Processed {idx + 1}/{len(dataset)} entries")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[Pref] Saved {len(results)} target entries to {output_path}")

    return results
