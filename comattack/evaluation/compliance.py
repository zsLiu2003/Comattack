
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..llm import BaseLLM
from .judge import ComplianceJudge
from .metrics import calculate_compliance_metrics

logger = logging.getLogger(__name__)


class BatchComplianceEvaluator:
    """
    Batch compliance evaluator.
    
    Encapsulates batch processing logic, separating target and judge calls.
    """
    
    def __init__(
        self,
        target_llm: BaseLLM,
        judge_llm: BaseLLM,
        max_workers: int = 8
    ):
        self.target_llm = target_llm
        self.judge_llm = judge_llm
        self.judge = ComplianceJudge(judge_llm)
        self.max_workers = max_workers
    
    def batch_target_inference(self, tasks: List[Dict]) -> Dict[str, str]:
        """
        Batch target LLM inference.
        
        Uses ThreadPoolExecutor for concurrent calls.
        task_id includes query_hash to ensure unique IDs per query.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for task in tasks:
                # Use query_hash to avoid collisions across different queries with same prompt_id + guardrail_index
                query_hash = hashlib.md5(task["query"].encode()).hexdigest()[:8]
                task_id = f"{task['prompt_id']}_{task['guardrail_index']}_{query_hash}"
                
                # If compressed result contains the full prompt, use it directly; otherwise pass separately
                if task.get("full_prompt"):
                    # Compressed result already contains system prompt + query; use as-is
                    future = executor.submit(
                        self.target_llm.generate,
                        prompt=task["full_prompt"],
                        system_prompt="",
                        temperature=0.0,
                        max_tokens=4096
                    )
                else:
                    future = executor.submit(
                        self.target_llm.generate,
                        prompt=task["query"],
                        system_prompt=task["system_prompt"],
                        temperature=0.0,
                        max_tokens=4096
                    )
                futures[future] = task_id
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Target inference"):
                task_id = futures[future]
                try:
                    response = future.result(timeout=120)
                    results[task_id] = response
                except Exception as e:
                    logger.warning(f"Target call failed: {e}")
                    results[task_id] = f"[ERROR: {str(e)}]"
        return results
    
    def batch_judge_inference(self, tasks: List[Dict], target_responses: Dict[str, str]) -> Dict[str, Dict]:
        """
        Batch judge LLM inference.
        
        After target responses are collected, batch-judge compliance.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for task in tasks:
                query_hash = hashlib.md5(task["query"].encode()).hexdigest()[:8]
                task_id = f"{task['prompt_id']}_{task['guardrail_index']}_{query_hash}"
                response = target_responses.get(task_id, "")
                
                # If target call failed, mark as error without calling judge
                if response.startswith("[ERROR:"):
                    results[task_id] = {
                        "compliant": None,
                        "verdict": "ERROR",
                        "confidence": 0.0,
                        "reasoning": response,
                        "severity": "unknown"
                    }
                    continue
                
                future = executor.submit(
                    self.judge.judge,
                    guardrail_text=task["guardrail_text"],
                    target_violation=task["target_violation"],
                    query=task["query"],
                    response=response
                )
                futures[future] = task_id
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Judge inference"):
                task_id = futures[future]
                try:
                    result = future.result(timeout=120)
                    results[task_id] = {
                        "compliant": result.is_compliant,
                        "verdict": result.verdict,
                        "confidence": result.confidence,
                        "reasoning": result.reasoning,
                        "severity": result.severity
                    }
                except Exception as e:
                    logger.warning(f"Judge call failed: {e}")
                    results[task_id] = {
                        "compliant": None,
                        "verdict": "ERROR",
                        "confidence": 0.0,
                        "reasoning": f"Error: {str(e)}",
                        "severity": "unknown"
                    }
        return results


def run_compliance_evaluation(
    queries_path: str,
    prompts_path: str,
    compression_path: str,
    output_path: str,
    target_llm: BaseLLM,
    judge_llm: BaseLLM,
    method_filter: Optional[List[str]] = None,
    max_workers: int = 8,
    allow_empty_prompt: bool = False
) -> Dict[str, Any]:
    """
    Run compliance evaluation: query + system prompt -> target LLM -> judge LLM
    
    Core pipeline function supporting multiple methods (original + compression methods).
    """
    # Load queries data
    logger.info("Loading queries...")
    with open(queries_path, 'r') as f:
        queries_data = json.load(f)
    queries = queries_data.get("queries", [])
    
    # Load system prompts from filtered_prompts file (format: {"prompt_ppls": [{"id": "...", "content": "..."}]})
    logger.info("Loading prompts...")
    with open(prompts_path, 'r') as f:
        prompts_data = json.load(f)
    original_prompts = {p["id"]: p.get("content", "") for p in prompts_data.get("prompt_ppls", [])}
    
    # Determine processing method based on compression_path availability
    compressed_prompts = {}
    compression_rates = {}
    all_methods = []
    is_prompt_only_mode = False
    
    if compression_path:
        # If compression path provided, process compressed methods only
        try:
            with open(compression_path, 'r') as f:
                compression_data = json.load(f)
            
            # Check if prompt-only mode
            is_prompt_only_mode = compression_data.get("metadata", {}).get("prompt_only", False)
            
            for method in ["llmlingua", "llmlingua2"]:
                if method not in compression_data:
                    continue
                
                for item in compression_data[method]:
                    prompt_id = str(item.get("prompt_id", ""))
                    rate = item.get("rate", "")
                    compressed_text = item.get("compressed_text", "")
                    
                    if prompt_id and compressed_text:
                        if prompt_id not in compressed_prompts:
                            compressed_prompts[prompt_id] = {}
                        key = f"{method}_{rate}"
                        
                        # In prompt-only mode, extract compressed prompt from compressed_text (strip query part)
                        if is_prompt_only_mode:
                            # compressed_text format: "{compressed_prompt}\n\nUser: {query}"
                            if "\n\nUser:" in compressed_text:
                                compressed_prompt = compressed_text.split("\n\nUser:")[0]
                            else:
                                compressed_prompt = compressed_text
                            compressed_prompts[prompt_id][key] = compressed_prompt
                        else:
                            # Original mode: use full compressed_text (already contains prompt + query)
                            compressed_prompts[prompt_id][key] = compressed_text
                        
                        compression_rates[key] = rate
                        if key not in all_methods:
                            all_methods.append(key)
        except Exception as e:
            logger.warning(f"Could not load compression data: {e}")
        
        all_methods = sorted(all_methods)
    else:
        # No compression path: process original only
        all_methods = ["original"]
    
    # If empty prompts allowed, add no_system_prompt method
    if allow_empty_prompt:
        all_methods.append("no_system_prompt")
    
    logger.info(f"Loaded {len(queries)} queries, {len(original_prompts)} prompts")
    logger.info(f"Methods: {all_methods}")
    
    # Initialize batch evaluator
    evaluator = BatchComplianceEvaluator(
        target_llm, judge_llm,
        max_workers=max_workers
    )
    
    # Store results for all methods, keyed by (prompt_id, guardrail_index, query_hash)
    all_evaluations = {}
    
    # Process each method (original + compression methods)
    for method in all_methods:
        # Apply method filter
        if method_filter and method not in ["original", "no_system_prompt"]:
            if not any(f in method for f in method_filter):
                logger.info(f"Skipping {method} (filtered)")
                continue
        
        logger.info(f"\nProcessing method: {method}")
        
        # Prepare tasks for current method
        tasks = []
        for query in queries:
            prompt_id = str(query.get("prompt_id", ""))
            
            # Get system prompt based on method type
            if method == "original":
                # Original method: use original system prompt + query
                system_prompt = original_prompts.get(prompt_id, "")
                query_text = query.get("adversarial_query", "")
                full_prompt = None
            elif method == "no_system_prompt":
                # No system prompt method: query only
                system_prompt = ""
                query_text = query.get("adversarial_query", "")
                full_prompt = None
            else:
                # Compression method
                compressed_prompt = compressed_prompts.get(prompt_id, {}).get(method, "")
                query_text = query.get("adversarial_query", "")
                
                if is_prompt_only_mode:
                    # Prompt-only mode: use compressed system prompt + original query
                    system_prompt = compressed_prompt
                    full_prompt = None
                else:
                    # Original mode: compressed_text already contains full prompt + query
                    full_prompt = compressed_prompt
                    system_prompt = ""
            
            # Skip if prompt is empty and empty prompts not allowed (except no_system_prompt)
            if method != "no_system_prompt" and not system_prompt and not full_prompt and not allow_empty_prompt:
                continue
            
            tasks.append({
                "prompt_id": prompt_id,
                "guardrail_index": query.get("guardrail_index", 0),
                "query": query_text,
                "guardrail_text": query.get("guardrail_text", ""),
                "target_violation": query.get("target_violation", ""),
                "system_prompt": system_prompt,
                "full_prompt": full_prompt,
            })
        
        if not tasks:
            continue
        
        logger.info(f"Tasks: {len(tasks)}")
        
        # Batch target LLM inference
        target_responses = evaluator.batch_target_inference(tasks)
        
        # Batch judge LLM inference for compliance
        judgments = evaluator.batch_judge_inference(tasks, target_responses)
        
        # Store results for current method, using query_hash for uniqueness
        for task in tasks:
            query_hash = hashlib.md5(task["query"].encode()).hexdigest()[:8]
            task_id = f"{task['prompt_id']}_{task['guardrail_index']}_{query_hash}"
            key = (task["prompt_id"], str(task["guardrail_index"]), query_hash)
            
            if key not in all_evaluations:
                all_evaluations[key] = {
                    "prompt_id": task["prompt_id"],
                    "guardrail_index": task["guardrail_index"],
                    "guardrail_text": task["guardrail_text"],
                    "query": task["query"],
                    "evaluations": {}
                }
            
            evaluation_result = {
                "response": target_responses.get(task_id, ""),
                "judgment": judgments.get(task_id, {})
            }
            
            # For compression methods, add compression rate info
            if method in compression_rates:
                evaluation_result["compression_rate"] = compression_rates[method]
            
            all_evaluations[key]["evaluations"][method] = evaluation_result
    
    # Convert to list format and compute statistics
    evaluations = list(all_evaluations.values())
    stats = calculate_compliance_metrics(evaluations)
    
    # Save results
    output = {
        "metadata": {
            "evaluation_date": datetime.now().isoformat(),
            "target_model": str(target_llm),
            "judge_model": str(judge_llm),
            "total_queries_loaded": len(queries),
            "total_queries": len(evaluations),
            "methods_evaluated": all_methods,
            "method_filter": method_filter
        },
        "statistics": stats,
        "evaluations": evaluations
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved results to: {output_path}")
    
    return output


def format_metrics_report(stats: Dict[str, Any]) -> str:
    """
    Format metrics as a human-readable report.
    """
    lines = [
        "=" * 50,
        "COMPLIANCE EVALUATION REPORT",
        "=" * 50,
        "",
        "OVERALL RESULTS:",
        f"  Total evaluated: {stats['overall'].get('total', 0)}",
        f"  Original compliance rate: {stats['overall'].get('original_compliance_rate', 0)*100:.1f}%",
        "",
    ]
    
    by_method = stats.get("by_method", {})
    if by_method:
        lines.append("BY COMPRESSION METHOD:")
        for method, method_stats in sorted(by_method.items()):
            rate = method_stats.get("compliance_rate", 0)
            total = method_stats.get("total", 0)
            lines.append(f"  {method}: {rate*100:.1f}% ({total} samples)")
        lines.append("")
    
    by_category = stats.get("by_category", {})
    if by_category:
        lines.append("BY GUARDRAIL CATEGORY:")
        for cat, cat_stats in sorted(by_category.items()):
            orig_rate = cat_stats.get("original_rate", 0)
            lines.append(f"  {cat}: {orig_rate*100:.1f}% original compliance")
        lines.append("")
    
    lines.append("=" * 50)
    return "\n".join(lines)
