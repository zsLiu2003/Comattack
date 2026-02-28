"""
Compliance Metrics Calculation
"""

from typing import Dict, List, Any
from collections import defaultdict


def calculate_compliance_metrics(evaluations: List[Dict]) -> Dict[str, Any]:
    """
    Calculate compliance statistics from evaluation results.
    
    Args:
        evaluations: List of evaluation results
        
    Returns:
        Statistics dictionary with rates by method, category, etc.
    """
    stats = {
        "overall": {},
        "by_method": {},
        "by_category": {},
        "by_high_level_category": {},
        "by_severity": {}
    }
    
    # Collect verdicts
    method_counts = defaultdict(lambda: {
        "compliant": 0, "violation": 0, "error": 0
    })
    
    category_counts = defaultdict(lambda: defaultdict(lambda: {
        "compliant": 0, "violation": 0
    }))
    
    high_level_counts = defaultdict(lambda: defaultdict(lambda: {
        "compliant": 0, "violation": 0
    }))
    
    severity_counts = defaultdict(int)
    
    for result in evaluations:
        category = result.get("guardrail_category", "unknown")
        high_level = result.get("high_level_category", "unknown")
        
        for method, eval_data in result.get("evaluations", {}).items():
            # Handle errors
            if "error" in eval_data:
                method_counts[method]["error"] += 1
                continue
            
            # Get judgment - handle both old (verdict) and new (compliant) formats
            judgment = eval_data.get("judgment", {})
            if isinstance(judgment, dict):
                # New format: compliant is boolean
                if "compliant" in judgment:
                    is_compliant = judgment.get("compliant")
                    if is_compliant is True:
                        verdict = "COMPLIANT"
                    elif is_compliant is False:
                        verdict = "VIOLATION"
                    else:
                        verdict = "ERROR"
                # Old format: verdict is string
                else:
                    verdict = judgment.get("verdict", "ERROR")
                severity = judgment.get("severity", "unknown")
            else:
                verdict = "ERROR"
                severity = "unknown"
            
            # Count verdicts
            if verdict == "COMPLIANT":
                method_counts[method]["compliant"] += 1
                category_counts[category][method]["compliant"] += 1
                high_level_counts[high_level][method]["compliant"] += 1
            elif verdict == "VIOLATION":
                method_counts[method]["violation"] += 1
                category_counts[category][method]["violation"] += 1
                high_level_counts[high_level][method]["violation"] += 1
                severity_counts[severity] += 1
            else:
                method_counts[method]["error"] += 1
    
    # Calculate rates by method
    for method, counts in method_counts.items():
        total = counts["compliant"] + counts["violation"]
        stats["by_method"][method] = {
            "compliant": counts["compliant"],
            "violation": counts["violation"],
            "error": counts["error"],
            "total": total,
            "compliance_rate": counts["compliant"] / total if total > 0 else 0,
            "violation_rate": counts["violation"] / total if total > 0 else 0
        }
    
    # Calculate rates by category
    for category, method_data in category_counts.items():
        stats["by_category"][category] = {}
        for method, counts in method_data.items():
            total = counts["compliant"] + counts["violation"]
            stats["by_category"][category][method] = {
                "compliance_rate": counts["compliant"] / total if total > 0 else 0,
                "violation_rate": counts["violation"] / total if total > 0 else 0,
                "total": total
            }
    
    # Calculate rates by high-level category
    for high_level, method_data in high_level_counts.items():
        stats["by_high_level_category"][high_level] = {}
        for method, counts in method_data.items():
            total = counts["compliant"] + counts["violation"]
            stats["by_high_level_category"][high_level][method] = {
                "compliance_rate": counts["compliant"] / total if total > 0 else 0,
                "violation_rate": counts["violation"] / total if total > 0 else 0,
                "total": total
            }
    
    # Severity distribution
    stats["by_severity"] = dict(severity_counts)
    
    # Overall stats
    total_evaluated = sum(d["total"] for d in stats["by_method"].values()) // max(1, len(stats["by_method"]))
    stats["overall"]["total"] = total_evaluated
    
    # Overall comparison (original vs compressed)
    if "original" in stats["by_method"]:
        original_rate = stats["by_method"]["original"]["compliance_rate"]
        stats["overall"]["original_compliance_rate"] = original_rate
        
        for method, data in stats["by_method"].items():
            if method != "original":
                compressed_rate = data["compliance_rate"]
                stats["overall"][f"{method}_compliance_rate"] = compressed_rate
                stats["overall"][f"{method}_compliance_drop"] = original_rate - compressed_rate
    
    return stats


def format_metrics_report(stats: Dict[str, Any]) -> str:
    """Format metrics as a readable report."""
    
    lines = []
    lines.append("=" * 60)
    lines.append("COMPLIANCE EVALUATION REPORT")
    lines.append("=" * 60)
    
    # By method
    lines.append("\n📈 Results by Method:")
    for method, data in stats.get("by_method", {}).items():
        lines.append(f"\n  {method}:")
        lines.append(f"    Compliance Rate: {data['compliance_rate']:.1%}")
        lines.append(f"    Violation Rate:  {data['violation_rate']:.1%}")
        lines.append(f"    Total: {data['total']} (errors: {data['error']})")
    
    # Compliance drop
    if stats.get("overall"):
        lines.append("\n📉 Compliance Drop (vs Original):")
        for key, value in stats["overall"].items():
            if "drop" in key:
                method = key.replace("_compliance_drop", "")
                lines.append(f"  {method}: {value:+.1%}")
    
    # By high-level category
    lines.append("\n📂 By High-Level Category:")
    for high_level, method_data in stats.get("by_high_level_category", {}).items():
        lines.append(f"\n  {high_level}:")
        for method, data in method_data.items():
            lines.append(f"    {method}: {data['compliance_rate']:.1%} (n={data['total']})")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)

