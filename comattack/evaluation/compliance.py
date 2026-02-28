"""
Compliance Evaluator - Evaluate guardrail compliance
"""

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
    
    理由：封装批处理逻辑，将 target 和 judge 的调用分离，代码更清晰
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
        批量调用 target LLM。
        
        理由：使用 ThreadPoolExecutor 并发调用，提高效率
        task_id 包含 query_hash 确保每个唯一 query 都有独立的 ID
        """
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for task in tasks:
                # 理由：使用 query_hash 确保同一 prompt_id + guardrail_index 的不同 query 不会冲突
                query_hash = hashlib.md5(task["query"].encode()).hexdigest()[:8]
                task_id = f"{task['prompt_id']}_{task['guardrail_index']}_{query_hash}"
                
                # 理由：如果压缩结果已经包含完整 prompt，直接使用；否则分开传递
                if task.get("full_prompt"):
                    # 压缩结果已经包含 system prompt + query，直接使用完整文本作为 prompt
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
        批量调用 judge LLM。
        
        理由：在 target 响应完成后，批量判断合规性
        """
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for task in tasks:
                query_hash = hashlib.md5(task["query"].encode()).hexdigest()[:8]
                task_id = f"{task['prompt_id']}_{task['guardrail_index']}_{query_hash}"
                response = target_responses.get(task_id, "")
                
                # 理由：如果 target 调用失败，直接标记为错误，不调用 judge
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
    运行合规性评估：query + system prompt -> target LLM -> judge LLM
    
    理由：核心流程函数，支持多种方法（original + 压缩方法）
    """
    # 理由：加载 queries 数据
    logger.info("Loading queries...")
    with open(queries_path, 'r') as f:
        queries_data = json.load(f)
    queries = queries_data.get("queries", [])
    
    # 理由：从 filtered_prompts 文件加载 system prompts（格式：{"prompt_ppls": [{"id": "...", "content": "..."}]}）
    logger.info("Loading prompts...")
    with open(prompts_path, 'r') as f:
        prompts_data = json.load(f)
    original_prompts = {p["id"]: p.get("content", "") for p in prompts_data.get("prompt_ppls", [])}
    
    # 理由：根据是否有 compression_path 决定处理方法
    compressed_prompts = {}
    compression_rates = {}  # 存储每个方法的压缩率
    all_methods = []
    is_prompt_only_mode = False  # 理由：存储是否是 prompt-only 模式
    
    if compression_path:
        # 理由：如果提供了压缩路径，只处理压缩方法
        try:
            with open(compression_path, 'r') as f:
                compression_data = json.load(f)
            
            # 理由：检查是否是 prompt-only 模式
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
                        
                        # 理由：如果是 prompt-only 模式，从 compressed_text 中提取压缩后的 prompt（去掉 query 部分）
                        if is_prompt_only_mode:
                            # compressed_text 格式: "{compressed_prompt}\n\nUser: {query}"
                            # 提取压缩后的 prompt 部分
                            if "\n\nUser:" in compressed_text:
                                compressed_prompt = compressed_text.split("\n\nUser:")[0]
                            else:
                                compressed_prompt = compressed_text
                            compressed_prompts[prompt_id][key] = compressed_prompt
                        else:
                            # 理由：原始模式，使用完整的 compressed_text（已包含 prompt + query）
                            compressed_prompts[prompt_id][key] = compressed_text
                        
                        compression_rates[key] = rate  # 存储压缩率
                        if key not in all_methods:
                            all_methods.append(key)
        except Exception as e:
            logger.warning(f"Could not load compression data: {e}")
        
        all_methods = sorted(all_methods)
    else:
        # 理由：如果没有压缩路径，只处理 original
        all_methods = ["original"]
    
    # 理由：如果允许空 prompt，添加 no_system_prompt 方法
    if allow_empty_prompt:
        all_methods.append("no_system_prompt")
    
    logger.info(f"Loaded {len(queries)} queries, {len(original_prompts)} prompts")
    logger.info(f"Methods: {all_methods}")
    
    # 理由：初始化批处理评估器
    evaluator = BatchComplianceEvaluator(
        target_llm, judge_llm,
        max_workers=max_workers
    )
    
    # 理由：存储所有方法的结果，key 为 (prompt_id, guardrail_index, query_hash)
    all_evaluations = {}
    
    # 理由：循环处理每个方法（original + 各种压缩方法）
    for method in all_methods:
        # 理由：应用方法过滤
        if method_filter and method not in ["original", "no_system_prompt"]:
            if not any(f in method for f in method_filter):
                logger.info(f"Skipping {method} (filtered)")
                continue
        
        logger.info(f"\nProcessing method: {method}")
        
        # 理由：为当前方法准备任务
        tasks = []
        for query in queries:
            prompt_id = str(query.get("prompt_id", ""))
            
            # 理由：根据方法类型获取 system prompt
            if method == "original":
                # 理由：original 方法使用原始 system prompt + query
                system_prompt = original_prompts.get(prompt_id, "")
                query_text = query.get("adversarial_query", "")
                full_prompt = None
            elif method == "no_system_prompt":
                # 理由：no_system_prompt 方法不使用 system prompt，只有 query
                system_prompt = ""
                query_text = query.get("adversarial_query", "")
                full_prompt = None
            else:
                # 理由：压缩方法
                compressed_prompt = compressed_prompts.get(prompt_id, {}).get(method, "")
                query_text = query.get("adversarial_query", "")
                
                if is_prompt_only_mode:
                    # 理由：prompt-only 模式：使用压缩后的 system prompt + 原始 query
                    system_prompt = compressed_prompt
                    full_prompt = None
                else:
                    # 理由：原始模式：compressed_text 已包含完整的 prompt + query
                    full_prompt = compressed_prompt
                    system_prompt = ""
            
            # 理由：如果 prompt 为空且不允许空 prompt，跳过（no_system_prompt 除外）
            if method != "no_system_prompt" and not system_prompt and not full_prompt and not allow_empty_prompt:
                continue
            
            tasks.append({
                "prompt_id": prompt_id,
                "guardrail_index": query.get("guardrail_index", 0),
                "query": query_text,
                "guardrail_text": query.get("guardrail_text", ""),
                "target_violation": query.get("target_violation", ""),
                "system_prompt": system_prompt,
                "full_prompt": full_prompt,  # 压缩后的完整 prompt
            })
        
        if not tasks:
            continue
        
        logger.info(f"Tasks: {len(tasks)}")
        
        # 理由：批量调用 target LLM 获取响应
        target_responses = evaluator.batch_target_inference(tasks)
        
        # 理由：批量调用 judge LLM 判断合规性
        judgments = evaluator.batch_judge_inference(tasks, target_responses)
        
        # 理由：存储当前方法的结果，使用 query_hash 确保唯一性
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
            
            # 理由：如果是压缩方法，添加压缩率信息
            if method in compression_rates:
                evaluation_result["compression_rate"] = compression_rates[method]
            
            all_evaluations[key]["evaluations"][method] = evaluation_result
    
    # 理由：转换为列表格式并计算统计指标
    evaluations = list(all_evaluations.values())
    stats = calculate_compliance_metrics(evaluations)
    
    # 理由：保存结果
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
    格式化指标报告。
    
    理由：将统计结果格式化为可读的文本报告
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
