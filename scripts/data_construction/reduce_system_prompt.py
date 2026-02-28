#!/usr/bin/env python3
"""
Reduce system prompts to: role opening + guardrail sentences.
Reads extracted_guardrails.json, outputs extracted_guardrails_reduced_filtered.json
directly in the format expected by generate_adversarial_queries_v2.py.

IMPORTANT: each guardrail sentence must appear verbatim in the output system_prompt
so that attack_manager.py's keyword_char_span_in_fulltext() can locate it.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
INPUT_JSON = PROJECT_ROOT / "data" / "extracted_guardrails.json"
OUTPUT_JSON = PROJECT_ROOT / "data" / "extracted_guardrails_reduced_filtered.json"

# How much of the role opening to keep (chars). Cut at last newline if mid-line.
BEGINNING_CHARS = 500


def reduce_prompt(system_prompt: str, guardrail_list: list) -> str:
    """
    Build reduced system prompt: role opening + guardrail sentences.

    The role opening is the first BEGINNING_CHARS characters (cut at last
    newline boundary to avoid mid-sentence cuts).

    Each guardrail sentence is appended on its own line, separated by a
    blank line from the opening. Sentences are deduplicated.
    """
    # --- role opening ---
    if len(system_prompt) <= BEGINNING_CHARS:
        beginning = system_prompt.rstrip()
    else:
        beginning = system_prompt[:BEGINNING_CHARS].rstrip()
        if "\n" in beginning:
            beginning = beginning.rsplit("\n", 1)[0]

    # --- guardrail sentences (deduplicated, preserve order) ---
    seen = set()
    sentences = []
    for g in guardrail_list:
        sent = g["sentence"].strip()
        if not sent or sent in seen:
            continue
        # Skip if sentence already appears in the beginning
        if sent in beginning:
            seen.add(sent)
            continue
        seen.add(sent)
        sentences.append(sent)

    if not sentences:
        return beginning

    return beginning + "\n\n" + "\n".join(sentences)


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise SystemExit(f"Unexpected format: root is {type(data)}, expected list")

    output = []
    total_guardrails = 0
    kept_guardrails = 0

    for entry in data:
        system_prompt = entry.get("system_prompt", "")
        guardrail_list = entry.get("guardrail_list", [])
        if not guardrail_list:
            continue

        reduced = reduce_prompt(system_prompt, guardrail_list)

        # Only keep guardrails whose sentence appears verbatim in the reduced prompt
        filtered_guardrails = []
        for g in guardrail_list:
            total_guardrails += 1
            if g["sentence"] in reduced:
                filtered_guardrails.append(g)
                kept_guardrails += 1

        if not filtered_guardrails:
            continue

        output.append({
            "system_prompt": reduced,
            "guardrail_list": filtered_guardrails,
        })

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total_g_out = sum(len(e["guardrail_list"]) for e in output)
    avg_len = sum(len(e["system_prompt"]) for e in output) / max(len(output), 1)
    print(f"Entries: {len(output)}")
    print(f"Guardrails: {total_g_out} (kept {kept_guardrails}/{total_guardrails})")
    print(f"Avg reduced prompt length: {avg_len:.0f} chars")
    print(f"Output: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
