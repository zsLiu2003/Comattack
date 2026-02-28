#!/usr/bin/env python3
"""
Extract guardrails from system prompt files in System_Prompt_Lib.
Output: one JSON file with format:
  [{ "system_prompt": "...", "guardrail_list": [ {"keyword": "...", "sentence": "..."}, ... ] }, ...]

Extraction strategy (v2 — tighter):
  1. Split text into sentence-level chunks.
  2. For each chunk, check if a prohibitive/obligatory keyword appears in
     a *command position* (near the start, not inside a conditional clause).
  3. Filter out false positives: quoted example responses, section headers,
     conditional descriptions, tool-behavior descriptions.
  4. Cap guardrails per entry to avoid over-extraction.
"""

import json
import re
from pathlib import Path
from typing import Optional

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
SYSTEM_PROMPT_LIB = PROJECT_ROOT / "System_Prompt_Lib"
OUTPUT_JSON = PROJECT_ROOT / "data" / "extracted_guardrails.json"

# ─── Config ───────────────────────────────────────────────────────────────────
TEXT_EXTENSIONS = {".md", ".txt"}
SKIP_NAMES = {"readme.md", "readme.txt", ".gitignore", "license", "claude.md"}
MAX_GUARDRAILS_PER_ENTRY = 30
MIN_SENTENCE_LEN = 20
MAX_SENTENCE_LEN = 400

# ─── Keyword patterns ────────────────────────────────────────────────────────
# Ordered longest-first so greedy match picks the most specific phrase.
KEYWORD_INDICATORS = re.compile(
    r"\b("
    r"under\s+no\s+circumstances|"
    r"do\s+not\s+attempt|do\s+not\s+ever|"
    r"must\s+never|must\s+not|must\s+always|"
    r"you\s+must\s+not|you\s+must\s+never|you\s+may\s+not|"
    r"you\s+cannot|you\s+should\s+not|you\s+should\s+never|"
    r"is\s+strictly\s+prohibited|is\s+strictly\s+forbidden|"
    r"is\s+not\s+allowed|are\s+not\s+allowed|"
    r"never\s+ever|"
    r"do\s+not|don't|never|"
    r"cannot|can't|"
    r"should\s+not|shouldn't|"
    r"will\s+not|won't|"
    r"not\s+allowed|not\s+permitted|"
    r"prohibited|forbidden|"
    r"always|must"
    r")\b",
    re.IGNORECASE,
)

# ─── False-positive filters ──────────────────────────────────────────────────

_CONDITIONAL_START = re.compile(
    r"^(?:if|when|where|while|unless|in\s+case|whenever|whether)\s",
    re.IGNORECASE,
)

_QUOTED_RESPONSE = re.compile(
    r"""^["'""\u201c]\s*(?:I\s+cannot|I'm\s+sorry|I\s+am\s+unable|I\s+can't|I\s+don't|Sorry)""",
    re.IGNORECASE,
)

_SECTION_HEADER = re.compile(r":\s*$")

_DESCRIPTIVE_SUBJ = re.compile(
    r"^(?:[-*•\d.)\s]*)"
    r"(?:the\s+|this\s+|these\s+|those\s+|that\s+|it\s+|its\s+|"
    r"past\s+|search\s+|tool|result|output|response|question|"
    r"information|content|data|file|document|message|conversation|"
    r"technical|external|additional|previous|current)\s*\w",
    re.IGNORECASE,
)


def normalize_whitespace(s: str) -> str:
    return " ".join(s.split()).strip()


def _strip_bullet(s: str) -> str:
    """Remove leading bullet markers: '- ', '* ', '• ', '1. ', '1) '."""
    return re.sub(r"^[\s]*[-*•]\s+|^[\s]*\d+[.)]\s+", "", s).strip()


def _strip_markdown(s: str) -> str:
    """Remove bold/italic markers."""
    return re.sub(r"\*{1,2}|_{1,2}", "", s).strip()


def _is_false_positive(sentence: str, keyword: str) -> bool:
    """Return True if this sentence is a false-positive guardrail."""
    stripped = _strip_bullet(sentence)
    stripped_md = _strip_markdown(stripped)

    # 1. Section header (short, ends with colon)
    if _SECTION_HEADER.search(stripped_md) and len(stripped_md) < 80:
        return True

    # 2. Quoted example response
    if _QUOTED_RESPONSE.search(stripped_md):
        return True

    # 3. Conditional clause where keyword is inside the condition body
    if _CONDITIONAL_START.match(stripped_md):
        comma_pos = stripped_md.find(",")
        if comma_pos == -1:
            # No comma → entire sentence is a conditional → not a guardrail
            return True
        kw_match = re.search(re.escape(keyword), stripped_md[:comma_pos], re.IGNORECASE)
        if kw_match:
            return True

    # 4. Descriptive subject (tool/result/question does X) — not a directive
    if _DESCRIPTIVE_SUBJ.match(stripped_md):
        kw_lower = keyword.lower().strip()
        first_50 = stripped_md[:50].lower()
        if kw_lower not in first_50:
            return True
        # "The assistant must not..." is OK
        if re.match(r"(?:the\s+)?assistant", stripped_md, re.IGNORECASE):
            return False
        # "This tool cannot..." etc. are descriptions
        if re.match(r"(?:this|these|those|that|it)\s", stripped_md, re.IGNORECASE):
            return True

    return False


def _keyword_in_command_position(sentence: str, keyword: str) -> bool:
    """
    Check that the keyword appears within the first ~80 chars of the sentence
    (after stripping bullets/markdown), indicating it's a directive.
    """
    stripped = _strip_markdown(_strip_bullet(sentence))
    m = re.search(re.escape(keyword), stripped, re.IGNORECASE)
    if not m:
        return False
    return m.start() < 80


def _keyword_from_sentence(sentence: str) -> Optional[str]:
    """Extract the first matching keyword phrase from the sentence (first 80 chars)."""
    stripped = _strip_markdown(_strip_bullet(sentence))
    m = KEYWORD_INDICATORS.search(stripped[:80])
    if m:
        return m.group(1).strip()
    return None


def extract_guardrail_pairs(text: str) -> list[dict]:
    """
    Extract guardrails as (keyword, sentence) pairs with strict filtering.
    """
    seen_norm = set()
    pairs = []

    def try_add(raw: str) -> None:
        sentence = normalize_whitespace(raw)
        if len(sentence) < MIN_SENTENCE_LEN:
            return
        norm_key = sentence.lower()
        if norm_key in seen_norm:
            return
        if len(sentence) > MAX_SENTENCE_LEN:
            cut = sentence[:MAX_SENTENCE_LEN]
            if "." in cut[MIN_SENTENCE_LEN:]:
                sentence = cut.rsplit(".", 1)[0] + "."
            else:
                sentence = cut

        keyword = _keyword_from_sentence(sentence)
        if keyword is None:
            return
        if not _keyword_in_command_position(sentence, keyword):
            return
        if _is_false_positive(sentence, keyword):
            return

        seen_norm.add(norm_key)
        pairs.append({"keyword": keyword, "sentence": sentence})

    # Split by newlines (catches bullet points)
    for line in text.split("\n"):
        line = line.strip()
        if line:
            try_add(line)

    # Split by sentence boundaries (catches multi-sentence lines)
    for chunk in re.split(r"(?<=[.!?])\s+", text):
        chunk = chunk.strip()
        if chunk:
            try_add(chunk)

    return pairs[:MAX_GUARDRAILS_PER_ENTRY]


# ─── File I/O ─────────────────────────────────────────────────────────────────

def read_prompt_from_file(filepath: Path) -> Optional[str]:
    """Read system prompt content from a file."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if not content or len(content.strip()) < 50:
        return None
    if filepath.suffix.lower() == ".json":
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                for key in ("content", "prompt", "system", "system_prompt", "text"):
                    if key in data and isinstance(data[key], str):
                        return data[key]
            return None
        except json.JSONDecodeError:
            return None
    return content


def collect_prompt_files(base: Path):
    """Yield (filepath, source_folder) for all prompt files under base."""
    if not base.exists():
        print(f"WARNING: {base} does not exist")
        return
    for top in sorted(base.iterdir()):
        if not top.is_dir() or top.name.startswith("."):
            continue
        for path in top.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            if path.name.lower() in SKIP_NAMES:
                continue
            yield path, top.name


def main():
    results = []
    seen_content_hashes = set()
    total_files = 0
    processed = 0
    total_guardrails = 0

    for filepath, source in collect_prompt_files(SYSTEM_PROMPT_LIB):
        total_files += 1
        content = read_prompt_from_file(filepath)
        if not content:
            continue
        h = hash(content[:2000])
        if h in seen_content_hashes:
            continue
        seen_content_hashes.add(h)

        guardrail_list = extract_guardrail_pairs(content)

        # Only include entries that have at least 1 guardrail
        if not guardrail_list:
            continue

        total_guardrails += len(guardrail_list)
        results.append({
            "system_prompt": content,
            "source_file": str(filepath.relative_to(SYSTEM_PROMPT_LIB)),
            "source_repo": source,
            "guardrail_list": guardrail_list,
        })
        processed += 1

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Scanned {total_files} files under {SYSTEM_PROMPT_LIB}")
    print(f"Entries with guardrails: {processed}")
    print(f"Total guardrails: {total_guardrails}")
    print(f"Avg guardrails/entry: {total_guardrails / max(processed, 1):.1f}")
    print(f"Output: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
