#!/usr/bin/env python3
"""
This script is used to generate adversarial queries for hand-written system prompts.
为每个 guardrail 生成 adversarial query（JSON 格式）。
输入：extracted_guardrails.json（system_prompt, guardrail_list: [{keyword, sentence}]）
LLM 输入：完整 system_prompt + guardrail；输出：JSON（adversarial_query, target_violation 等）。
输出：同结构，每个 guardrail 增加 adversarial_query（字符串）和 adversarial_generation（解析后的 JSON 对象）。
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
from pathlib import Path

from comattack.llm import create_llm, LLMConfig, BaseLLM

logger = logging.getLogger(__name__)

# Default path: data/adversarial_query_generation_system_prompt.txt (relative to project root)
DEFAULT_SYSTEM_PROMPT_FILE = Path(__file__).parent.parent / "data" / "adversarial_query_generation_system_prompt.txt"


def load_system_prompt(path: Optional[Path] = None) -> str:
    """Load system prompt from file. If path is None, use DEFAULT_SYSTEM_PROMPT_FILE."""
    p = path if path is not None else DEFAULT_SYSTEM_PROMPT_FILE
    if not p.exists():
        raise FileNotFoundError(
            f"System prompt file not found: {p}. "
            "Create data/adversarial_query_generation_system_prompt.txt or pass --system-prompt-file."
        )
    return p.read_text(encoding="utf-8").strip()


def parse_json_from_response(response: str) -> Optional[dict]:
    """Extract and parse a JSON object from LLM response (handles <think>, markdown, etc.)."""
    text = response.strip()
    # Remove <think>...</think> blocks (e.g. Qwen3 thinking)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()
    # Remove markdown code blocks
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if m:
            text = m.group(1)
        else:
            # Try first {...} span
            start = text.find("{")
            if start != -1:
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            text = text[start : i + 1]
                            break
    # Find first { ... } if not already a single object
    if not text.strip().startswith("{"):
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            text = m.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def generate_one_query(
    llm: BaseLLM,
    task_system_prompt: str,
    entry_system_prompt: str,
    guardrail_sentence: str,
    max_tokens: int = 4096,
) -> tuple[str, dict]:
    """
    为单个 guardrail 生成 adversarial query（JSON）。
    输入：完整 entry system_prompt + guardrail sentence。
    返回：(adversarial_query 字符串, 解析后的完整 JSON 对象；解析失败时 query 为空串，obj 为 {}).
    """
    user_prompt = f"""**Entire system prompt**: {entry_system_prompt[:8000]}{"..." if len(entry_system_prompt) > 8000 else ""}

**Guardrail**: {guardrail_sentence}"""

    try:
        response = llm.generate(
            prompt=user_prompt,
            system_prompt=task_system_prompt,
            temperature=0.7,
            max_tokens=max_tokens,
            enable_thinking=True,
        )
        obj = parse_json_from_response(response)
        if not obj or not isinstance(obj, dict):
            return "", {}
        query = (obj.get("adversarial_query") or "").strip()
        if not query or len(query) > 2000:
            return "", obj
        return query, obj
    except Exception as e:
        logger.warning("Generate query failed: %s", e)
        return "", {}


def run_one(
    llm: BaseLLM,
    task_system_prompt: str,
    entry_system_prompt: str,
    entry_idx: int,
    guardrail_idx: int,
    guardrail: dict,
) -> tuple[int, int, dict]:
    """单个任务：返回 (entry_idx, guardrail_idx, guardrail + adversarial_query + adversarial_generation)."""
    sentence = guardrail.get("sentence", "")
    if not sentence:
        return entry_idx, guardrail_idx, {
            **guardrail,
            "adversarial_query": "",
            "adversarial_generation": {},
        }
    query, gen = generate_one_query(
        llm, task_system_prompt, entry_system_prompt, sentence
    )
    return entry_idx, guardrail_idx, {
        **guardrail,
        "adversarial_query": query,
        "adversarial_generation": gen,
    }


def generate_adversarial_queries(
    data: list,
    model: BaseLLM,
    system_prompt: str,
    max_workers: int = 16,
    sample: Optional[int] = None,
) -> list:
    """
    为 data 中每个 entry 的每个 guardrail 生成 adversarial query，并行执行。

    data: list of {system_prompt, guardrail_list: [{keyword, sentence}], ...}
    sample: 若指定，只处理前 sample 个 entry（用于测试）
    """
    if sample is not None and sample < len(data):
        data = data[:sample]

    # 展平为 (entry_idx, guardrail_idx, entry, guardrail)；传入完整 entry system_prompt
    tasks = []
    for ei, entry in enumerate(data):
        entry_sys = entry.get("system_prompt") or ""
        for gi, g in enumerate(entry.get("guardrail_list") or []):
            tasks.append((ei, gi, entry_sys, g))

    results = {}  # (ei, gi) -> guardrail_with_query_and_generation
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_one, model, system_prompt, entry_sys, ei, gi, g
            ): (ei, gi, g)
            for ei, gi, entry_sys, g in tasks
        }
        for future in as_completed(futures):
            ei, gi, g = futures[future]
            try:
                _, _, guardrail_with_result = future.result()
                results[(ei, gi)] = guardrail_with_result
            except Exception as e:
                logger.warning("Task failed (ei=%s gi=%s): %s", ei, gi, e)
                results[(ei, gi)] = {
                    **g,
                    "adversarial_query": "",
                    "adversarial_generation": {},
                }

    # 按 entry 重组
    out = []
    for ei, entry in enumerate(data):
        guardrail_list_old = entry.get("guardrail_list") or []
        guardrail_list_new = [
            results.get(
                (ei, gi),
                {**g, "adversarial_query": "", "adversarial_generation": {}},
            )
            for gi, g in enumerate(guardrail_list_old)
        ]
        out.append({
            **{k: v for k, v in entry.items() if k != "guardrail_list"},
            "guardrail_list": guardrail_list_new,
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="为每个 guardrail 生成 adversarial query")
    parser.add_argument("--input_path", required=True, help="输入 JSON（extracted_guardrails 格式）")
    parser.add_argument("--output_path", required=True, help="输出 JSON")
    parser.add_argument("--model", required=True, help="用于生成 query 的模型名")
    parser.add_argument("--target-url", default="http://localhost:8000/v1", help="模型 API URL")
    parser.add_argument("--system-prompt-file", default=None, help="生成 query 的 system prompt 文件（默认: data/adversarial_query_generation_system_prompt.txt）")
    parser.add_argument("--max-workers", type=int, default=16, help="并行 worker 数")
    parser.add_argument("--sample", type=int, default=None, help="只处理前 N 个 entry（测试用）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise SystemExit("input_path 应为 JSON 数组")

    system_prompt_path = Path(args.system_prompt_file) if args.system_prompt_file else None
    system_prompt = load_system_prompt(system_prompt_path)

    llm = create_llm(LLMConfig(
        model_name=args.model,
        provider="server",
        model_type="auxiliary",
        extra={"base_url": args.target_url, "api_key": "EMPTY"},
    ))

    print("=" * 60)
    print("Generate adversarial queries per guardrail")
    print("=" * 60)
    print(f"System prompt: {system_prompt_path or DEFAULT_SYSTEM_PROMPT_FILE}")
    print(f"Entries: {len(data)}")
    if args.sample:
        print(f"Sample: first {args.sample}")
    print(f"Workers: {args.max_workers}")
    print("=" * 60)

    results = generate_adversarial_queries(
        data=data,
        model=llm,
        system_prompt=system_prompt,
        max_workers=args.max_workers,
        sample=args.sample,
    )

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_with_query = sum(
        1
        for e in results
        for g in e.get("guardrail_list", [])
        if (g.get("adversarial_query") or "").strip()
    )
    n_total = sum(len(e.get("guardrail_list") or []) for e in results)
    print(f"Done. Guardrails with query: {n_with_query}/{n_total}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
