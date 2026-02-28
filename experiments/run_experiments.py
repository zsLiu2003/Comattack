#!/usr/bin/env python3
"""

Experiments:
1. Compression baseline comparison (P0.5)
2. Span-level preservation analysis (P0.2, P0.3)
3. Catastrophic edit detection (P1.3)
4. Behavioral evaluation (P0.4)
5. Mitigation effectiveness (P1.6)

Output:
- Results tables in markdown format
- Figures for paper
- Raw data in JSON format
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict
import numpy as np
from tqdm import tqdm

from .config import ExperimentConfig, ConstraintType
from .metrics import (
    CompressionRatioCalculator,
    SpanPreservationCalculator,
    CatastrophicEditDetector,
    ResultsReporter
)
from .baselines import BaselineManager
from .gold_annotation import (
    StratifiedSampler,
    AutomaticSpanExtractor,
    GoldEvaluator
)
from behavioral_eval import (
    ModelRunner,
    BehavioralEvaluator,
    QueryGenerator,
    BehavioralTestCase,
    TestQuery
)
from mitigations import MitigationManager, MitigationEvaluator


# =============================================================================
# Data Loading
# =============================================================================

def load_prompts(config: ExperimentConfig) -> List[Dict]:
    """Load system prompts from consolidated data."""
    data_path = os.path.join(config.data_dir, "consolidated_guardrails.json")
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found: {data_path}")
        return []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = []
    for prompt_data in data.get('prompts', []):
        content = prompt_data.get('content', '')
        if content and len(content) > 100:
            prompts.append({
                'prompt_id': prompt_data.get('file_name', ''),
                'content': content,
                'provider': prompt_data.get('provider', 'Unknown'),
                'model': prompt_data.get('model', 'unknown'),
                'guardrail_categories': prompt_data.get('guardrail_categories', [])
            })
    
    print(f"[LOADED] {len(prompts)} system prompts")
    return prompts


# =============================================================================
# Experiment 1: Baseline Comparison (P0.5)
# =============================================================================

def run_baseline_comparison(
    prompts: List[Dict],
    config: ExperimentConfig,
    output_dir: str
) -> Dict:
    """
    Compare compression methods against baselines.
    
    Shows that learned compression is meaningfully different from:
    - Truncation
    - Random deletion
    - Structure-preserving
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Baseline Comparison (P0.5)")
    print("=" * 60)
    
    baseline_manager = BaselineManager(config.downstream_models[0])
    span_calc = SpanPreservationCalculator()
    extractor = AutomaticSpanExtractor()
    
    results = {
        'by_method': {},
        'by_ratio': {}
    }
    
    # Sample prompts for efficiency
    sample_prompts = prompts[:100]
    
    for ratio in config.compression_ratios:
        print(f"\n[RATIO {ratio}]")
        ratio_results = {}
        
        for prompt in tqdm(sample_prompts, desc=f"Processing ratio {ratio}"):
            content = prompt['content']
            spans = extractor.extract(content)
            
            # Run all baselines
            baseline_results = baseline_manager.run_all(content, ratio)
            
            for method, result in baseline_results.items():
                if method not in ratio_results:
                    ratio_results[method] = {
                        'preservation_results': [],
                        'compression_ratios': []
                    }
                
                # Calculate span preservation
                pres_result = span_calc.calculate_prompt_level(spans, result.compressed)
                ratio_results[method]['preservation_results'].append(pres_result)
                ratio_results[method]['compression_ratios'].append(result.compression_ratio)
        
        # Aggregate by method
        for method, method_data in ratio_results.items():
            macro_micro = span_calc.calculate_macro_micro(method_data['preservation_results'])
            
            if method not in results['by_method']:
                results['by_method'][method] = {}
            
            results['by_method'][method][ratio] = {
                'macro_preservation': macro_micro['macro']['rate'],
                'micro_preservation': macro_micro['micro']['rate'],
                'mean_cr': np.mean(method_data['compression_ratios']),
                'std_cr': np.std(method_data['compression_ratios'])
            }
        
        results['by_ratio'][ratio] = ratio_results
    
    # Save results
    output_path = os.path.join(output_dir, "baseline_comparison.json")
    with open(output_path, 'w') as f:
        json.dump(results['by_method'], f, indent=2, default=float)
    print(f"\n[SAVED] {output_path}")
    
    # Generate table
    print("\n" + "=" * 40)
    print("BASELINE COMPARISON TABLE")
    print("=" * 40)
    
    print("\n| Method | Ratio | Macro Pres. | Micro Pres. | Actual CR |")
    print("|--------|-------|-------------|-------------|-----------|")
    
    for method in results['by_method']:
        for ratio in config.compression_ratios:
            data = results['by_method'][method][ratio]
            print(f"| {method} | {ratio} | {data['macro_preservation']:.3f} | "
                  f"{data['micro_preservation']:.3f} | {data['mean_cr']:.3f} |")
    
    return results


# =============================================================================
# Experiment 2: Catastrophic Edit Detection (P1.3)
# =============================================================================

def run_catastrophic_edit_analysis(
    prompts: List[Dict],
    config: ExperimentConfig,
    output_dir: str
) -> Dict:
    """
    Detect and analyze catastrophic edits:
    - Negation flips
    - Obligation weakening
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Catastrophic Edit Detection (P1.3)")
    print("=" * 60)
    
    detector = CatastrophicEditDetector()
    baseline_manager = BaselineManager(config.downstream_models[0])
    
    results = {}
    
    # Sample prompts
    sample_prompts = prompts[:200]
    
    for ratio in config.compression_ratios:
        print(f"\n[RATIO {ratio}]")
        
        pairs = []
        for prompt in tqdm(sample_prompts, desc=f"Compressing at {ratio}"):
            content = prompt['content']
            
            # Use truncate_first as representative compression
            # In full experiment, test all compressors
            result = baseline_manager.get_baseline('truncate_first').compress(content, ratio)
            pairs.append((content, result.compressed))
        
        # Analyze catastrophic edits
        analysis = detector.batch_analyze(pairs)
        
        results[ratio] = {
            'negation_flip_rate': analysis['negation_flip_rate'],
            'obligation_weakening_rate': analysis['obligation_weakening_rate'],
            'critical_edit_rate': analysis['critical_edit_rate'],
            'total_negation_flips': analysis['total_negation_flips'],
            'total_obligation_weakenings': analysis['total_obligation_weakenings']
        }
        
        print(f"  Negation Flip Rate: {analysis['negation_flip_rate']:.3f}")
        print(f"  Obligation Weakening Rate: {analysis['obligation_weakening_rate']:.3f}")
        print(f"  Critical Edit Rate: {analysis['critical_edit_rate']:.3f}")
    
    # Save results
    output_path = os.path.join(output_dir, "catastrophic_edits.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[SAVED] {output_path}")
    
    return results


# =============================================================================
# Experiment 3: Mitigation Effectiveness (P1.6)
# =============================================================================

def run_mitigation_experiment(
    prompts: List[Dict],
    config: ExperimentConfig,
    output_dir: str
) -> Dict:
    """
    Evaluate mitigation effectiveness.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Mitigation Effectiveness (P1.6)")
    print("=" * 60)
    
    mitigation_manager = MitigationManager()
    mitigation_evaluator = MitigationEvaluator()
    baseline_manager = BaselineManager(config.downstream_models[0])
    
    results = {
        'by_method': {},
        'by_ratio': {}
    }
    
    # Sample prompts
    sample_prompts = prompts[:100]
    
    for ratio in config.compression_ratios:
        print(f"\n[RATIO {ratio}]")
        
        ratio_results = {
            'no_mitigation': [],
            'guardrail_locking': [],
            'two_stage': []
        }
        
        compressor = baseline_manager.get_baseline('truncate_first')
        
        for prompt in tqdm(sample_prompts, desc=f"Testing mitigations at {ratio}"):
            content = prompt['content']
            
            # No mitigation
            baseline_result = compressor.compress(content, ratio)
            eval_no_mit = mitigation_evaluator.evaluate(content, baseline_result)
            eval_no_mit['method'] = 'no_mitigation'
            ratio_results['no_mitigation'].append(eval_no_mit)
            
            # With mitigations
            mit_results = mitigation_manager.run_all(content, compressor, ratio)
            
            for mit_name, mit_result in mit_results.items():
                if mit_name in ratio_results:
                    eval_mit = mitigation_evaluator.evaluate(content, mit_result)
                    ratio_results[mit_name].append(eval_mit)
        
        # Aggregate
        for method, method_results in ratio_results.items():
            if method not in results['by_method']:
                results['by_method'][method] = {}
            
            results['by_method'][method][ratio] = {
                'mean_tokens_saved': np.mean([r['tokens_saved'] for r in method_results]),
                'mean_preservation': np.mean([r['preservation_rate'] for r in method_results]),
                'mean_cr': np.mean([r['compression_ratio'] for r in method_results])
            }
        
        results['by_ratio'][ratio] = ratio_results
    
    # Save results
    output_path = os.path.join(output_dir, "mitigation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results['by_method'], f, indent=2, default=float)
    print(f"\n[SAVED] {output_path}")
    
    # Print table
    print("\n" + "=" * 40)
    print("MITIGATION COMPARISON TABLE")
    print("=" * 40)
    
    print("\n| Method | Ratio | Tokens Saved | Preservation | CR |")
    print("|--------|-------|--------------|--------------|------|")
    
    for method in results['by_method']:
        for ratio in config.compression_ratios:
            data = results['by_method'][method][ratio]
            print(f"| {method} | {ratio} | {data['mean_tokens_saved']:.1f} | "
                  f"{data['mean_preservation']:.3f} | {data['mean_cr']:.3f} |")
    
    return results


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_all_experiments(config: ExperimentConfig):
    """Run all experiments."""
    print("=" * 70)
    print("EXPERIMENT SUITE")
    print("System Prompt Compression Safety Analysis")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    
    # Create output directory
    output_dir = os.path.join(config.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load data
    prompts = load_prompts(config)
    
    if not prompts:
        print("[ERROR] No prompts loaded. Exiting.")
        return
    
    # Run experiments
    all_results = {}
    
    # Experiment 1: Baseline Comparison
    all_results['baseline_comparison'] = run_baseline_comparison(
        prompts, config, output_dir
    )
    
    # Experiment 2: Catastrophic Edits
    all_results['catastrophic_edits'] = run_catastrophic_edit_analysis(
        prompts, config, output_dir
    )
    
    # Experiment 3: Mitigations
    all_results['mitigations'] = run_mitigation_experiment(
        prompts, config, output_dir
    )
    
    # Save all results
    summary_path = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'config': {
                'compression_ratios': config.compression_ratios,
                'downstream_models': config.downstream_models,
                'n_prompts': len(prompts)
            },
            'results': {k: str(type(v)) for k, v in all_results.items()}
        }, f, indent=2)
    
    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"End time: {datetime.now().isoformat()}")


def main():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument('--ratios', nargs='+', type=float, default=[0.4, 0.6, 0.8],
                        help='Compression ratios to test')
    parser.add_argument('--sample', type=int, default=None,
                        help='Number of prompts to sample')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'baseline', 'catastrophic', 'mitigation'],
                        help='Which experiment to run')
    
    args = parser.parse_args()
    
    config = ExperimentConfig()
    config.compression_ratios = args.ratios
    
    run_all_experiments(config)


if __name__ == "__main__":
    main()

