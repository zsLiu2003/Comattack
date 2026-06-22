
"""
Evaluate Guardrail Compliance
=============================

Two modes:
- Server: Use utils/llm_provider.py (for remote API or vLLM server)
- Offline: Use vLLM to load HuggingFace models directly
"""

import argparse
import logging
from pathlib import Path

import sys
from pathlib import Path

from comattack.utils import load_config, setup_logging
from comattack.llm import create_llm, LLMConfig
from comattack.evaluation import run_compliance_evaluation
from comattack.evaluation.metrics import format_metrics_report


def main():
 parser = argparse.ArgumentParser(description="Evaluate guardrail compliance")
 
 script_dir = Path(__file__).parent.parent
 parser.add_argument("--queries", default=str(script_dir / "results/phase3.5/adversarial_queries.json"))
 parser.add_argument("--prompts", default=str(script_dir / "results/phase1/hierarchical_categories.json"))
 parser.add_argument("--compression", default=str(script_dir / "results/phase2/extractive_compression.json"))
 parser.add_argument("--output", default=str(script_dir / "results/phase3.5/compliance_evaluation.json"))
 
 parser.add_argument("--target", default="Qwen/Qwen3-8B", help="Target model")
 parser.add_argument("--judge", default="gpt-5.2", help="Judge model (closed-source)")
 
 parser.add_argument("--provider", default="auto", 
 choices=["auto", "server", "offline"],
 help="Provider: auto, server (vLLM API), offline (vLLM loads HF)")
 
 parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
 parser.add_argument("--gpu-memory", type=float, default=0.9, help="GPU memory utilization")
 parser.add_argument("--max-model-len", type=int, default=8192, help="Max context length")
 parser.add_argument("--quantization", default=None, choices=[None, "awq", "gptq"])
 
 parser.add_argument("--sample", type=int, default=None, help="Sample size")
 parser.add_argument("--no-skip", action="store_true", help="Don't skip existing")
 parser.add_argument("--debug", action="store_true", help="Debug logging")
 
 args = parser.parse_args()
 
 setup_logging(debug=args.debug)
 config = load_config()
 api_config = config.get("api", {})
 
 print(f"\nTarget model: {args.target}")
 
 target_config = LLMConfig(
 model_name=args.target,
 provider=args.provider,
 model_type="target",
 tensor_parallel_size=args.tp,
 gpu_memory_utilization=args.gpu_memory,
 max_model_len=args.max_model_len,
 quantization=args.quantization,
 )
 target_llm = create_llm(target_config)
 
 print(f"\n Judge model: {args.judge}")
 judge_config = LLMConfig(
 model_name=args.judge,
 provider="server",
 model_type="judge",
 )
 judge_llm = create_llm(judge_config)
 
 print(f"\nRunning evaluation...")
 results = run_compliance_evaluation(
 queries_path=args.queries,
 prompts_path=args.prompts,
 compression_path=args.compression,
 output_path=args.output,
 target_llm=target_llm,
 judge_llm=judge_llm,
 sample_size=args.sample,
 skip_existing=not args.no_skip
 )
 
 report = format_metrics_report(results["statistics"])
 print(report)
 
 print(f"Results saved to: {args.output}")


if __name__ == "__main__":
 main()

