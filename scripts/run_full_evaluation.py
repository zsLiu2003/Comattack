#!/usr/bin/env python3
"""Run compliance evaluation."""

import argparse
import logging
from pathlib import Path

from comattack.llm import create_llm, LLMConfig
from comattack.evaluation import run_compliance_evaluation, format_metrics_report


def main():
    parser = argparse.ArgumentParser(description="Compliance evaluation")
    
    # Required paths
    parser.add_argument("--queries", required=True, help="Path to queries JSON")
    parser.add_argument("--prompts", required=True, help="Path to prompts JSON")
    parser.add_argument("--output", required=True, help="Output path")
    
    # Optional
    parser.add_argument("--compression", default="", help="Path to compression results (optional)")
    parser.add_argument("--methods", default=None, help="Method filter (comma-separated)")
    parser.add_argument("--allow-empty-prompt", action="store_true", help="Allow empty prompts")
    parser.add_argument("--max-workers", type=int, default=16, help="Max workers")
    
    # Models (required for LLM setup)
    parser.add_argument("--model", required=True, help="Target model")
    parser.add_argument("--judge", default="Qwen/Qwen3-32B", help="Judge model")
    parser.add_argument("--target-url", default="http://localhost:8000/v1", help="Target model URL")
    parser.add_argument("--judge-url", default="http://localhost:8001/v1", help="Judge model URL")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("COMPLIANCE EVALUATION")
    print("=" * 60)
    print(f"Queries: {args.queries}")
    print(f"Prompts: {args.prompts}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Create LLMs
    target_llm = create_llm(LLMConfig(
        model_name=args.model,
        provider="server",
        model_type="target",
        extra={"base_url": args.target_url, "api_key": "EMPTY"}
    ))
    
    judge_llm = create_llm(LLMConfig(
        model_name=args.judge,
        provider="server",
        model_type="judge",
        extra={"base_url": args.judge_url, "api_key": "EMPTY"}
    ))
    
    # Parse method filter
    # method_filter = [m.strip() for m in args.methods.split(",")] if args.methods else None
    
    # Run evaluation
    results = run_compliance_evaluation(
        queries_path=args.queries,
        prompts_path=args.prompts,
        compression_path=args.compression,
        output_path=args.output,
        target_llm=target_llm,
        judge_llm=judge_llm,
        max_workers=args.max_workers,
        allow_empty_prompt=args.allow_empty_prompt
    )
    
    # Print report
    print("\n" + "=" * 60)
    print(format_metrics_report(results["statistics"]))
    print(f"\n✅ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
