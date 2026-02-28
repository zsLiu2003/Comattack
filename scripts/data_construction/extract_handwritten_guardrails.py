#!/usr/bin/env python3
"""
Re-extract guardrails from handwritten system prompts using the same
regex-based extraction logic as extract_guardrails.py (leaked prompts).

This replaces the paraphrased guardrail summaries from the normster/SystemCheck
dataset with verbatim sentences extracted from the actual system prompt text,
ensuring keyword_char_span_in_fulltext() can locate them.

Post-processing ensures:
  1. Every sentence is a verbatim substring of the system prompt.
  2. Every keyword appears in its sentence (case-insensitive).

Input:  data/handwritten_system_prompt_guardrails.json
        (or any JSON with [{system_prompt, ...}, ...])
Output: data/handwritten_system_prompt_extracted_guardrails.json
        [{system_prompt, guardrail_list: [{keyword, sentence}, ...]}, ...]
"""

import sys
import re
import json
import argparse
from pathlib import Path

# Import from sibling script in same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_guardrails import extract_guardrail_pairs, KEYWORD_INDICATORS


def _strip_markdown(s: str) -> str:
    return re.sub(r"\*{1,2}|_{1,2}", "", s).strip()


def _find_verbatim_line(system_prompt: str, sentence: str) -> str:
    """
    Find the actual verbatim line in system_prompt that matches `sentence`.

    extract_guardrail_pairs() applies normalize_whitespace(), which may differ
    from the original whitespace.  We recover the original by scanning each line.
    """
    if sentence in system_prompt:
        return sentence

    # Try markdown-stripped version
    sent_clean = _strip_markdown(sentence)
    if sent_clean in system_prompt:
        return sent_clean

    # Scan lines: find the one whose normalized form matches
    norm = " ".join(sentence.split()).lower()
    for line in system_prompt.split("\n"):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if " ".join(line_stripped.split()).lower() == norm:
            return line_stripped
        # Also try markdown-stripped comparison
        line_clean = _strip_markdown(line_stripped)
        if " ".join(line_clean.split()).lower() == norm:
            return line_stripped

    return ""


def _fix_keyword_in_sentence(keyword: str, sentence: str) -> str:
    """
    The keyword was extracted from markdown-stripped text, so it may not
    appear verbatim in the (possibly markdown-containing) sentence.
    Return the keyword as it actually appears in the sentence, or "" if not found.
    """
    if keyword.lower() in sentence.lower():
        return keyword

    # Try finding the keyword with optional markdown markers between words
    # e.g. keyword="MUST NOT" should match "**MUST NOT**" or "**MUST** NOT"
    words = keyword.split()
    if len(words) > 1:
        pattern = r"\**\s*".join(re.escape(w) for w in words)
        pattern = r"\**" + pattern + r"\**"
        m = re.search(pattern, sentence, re.IGNORECASE)
        if m:
            return m.group(0)

    return ""


def validate_guardrails(system_prompt, guardrail_list):
    """
    Post-process extracted guardrails:
      - Ensure sentence is a verbatim substring of system_prompt
      - Ensure keyword appears in sentence
    Returns only valid guardrails.
    """
    valid = []
    for g in guardrail_list:
        sent = g["sentence"]
        kw = g["keyword"]

        # 1. Find verbatim sentence
        verbatim = _find_verbatim_line(system_prompt, sent)
        if not verbatim or verbatim not in system_prompt:
            continue

        # 2. Fix keyword to match the verbatim sentence
        fixed_kw = _fix_keyword_in_sentence(kw, verbatim)
        if not fixed_kw:
            continue

        valid.append({"keyword": fixed_kw, "sentence": verbatim})

    # Deduplicate by (keyword_lower, sentence)
    seen = set()
    deduped = []
    for g in valid:
        key = (g["keyword"].lower(), g["sentence"])
        if key not in seen:
            seen.add(key)
            deduped.append(g)

    return deduped


def main():
    proj = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Extract verbatim guardrails from handwritten system prompts",
    )
    parser.add_argument(
        "--input", type=str,
        default=str(proj / "data" / "handwritten_system_prompt_guardrails.json"),
        help="Input JSON with system_prompt entries",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(proj / "data" / "handwritten_system_prompt_extracted_guardrails.json"),
        help="Output JSON with extracted guardrails",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = []
    total_raw = 0
    total_valid = 0

    for entry in data:
        system_prompt = entry.get("system_prompt", "")
        if not system_prompt:
            continue

        raw_guardrails = extract_guardrail_pairs(system_prompt)
        total_raw += len(raw_guardrails)

        guardrail_list = validate_guardrails(system_prompt, raw_guardrails)
        if not guardrail_list:
            continue

        total_valid += len(guardrail_list)
        output.append({
            "system_prompt": system_prompt,
            "guardrail_list": guardrail_list,
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Entries with guardrails: {len(output)}/{len(data)}")
    print(f"Raw extracted: {total_raw}  →  Valid (verbatim): {total_valid}")
    print(f"Avg guardrails/entry: {total_valid / max(len(output), 1):.1f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
