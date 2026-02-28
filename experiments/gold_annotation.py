"""
ICML Gold Annotation Framework (P0.3)
=====================================

Create a gold annotation set for guardrail spans + types.

Protocol:
- Sample 250-500 system prompts (stratified by provider/source + length)
- Two annotators per prompt + adjudication
- Annotate spans and type labels (multi-label if needed)

Report:
- Inter-annotator agreement (Krippendorff's α or Cohen's κ)
- Span-level Precision/Recall/F1 for automatic extractor
- Error analysis: common false positives/negatives
"""

import json
import random
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
import re

from config import ConstraintType, ConstraintSpan


# =============================================================================
# Annotation Data Structures
# =============================================================================

@dataclass
class SpanAnnotation:
    """A single span annotation."""
    text: str
    start: int
    end: int
    constraint_type: str  # One of the 7 types
    annotator_id: str
    confidence: float = 1.0
    notes: str = ""
    
    # For boundary constraints (P1.1)
    allowed_scope: str = ""
    disallowed_scope: str = ""


@dataclass
class PromptAnnotation:
    """Full annotation for a system prompt."""
    prompt_id: str
    prompt_text: str
    spans: List[SpanAnnotation] = field(default_factory=list)
    annotator_id: str = ""
    annotation_time_seconds: float = 0.0
    
    # Metadata
    provider: str = ""
    length_bin: str = ""  # short, medium, long


@dataclass
class AdjudicatedAnnotation:
    """Adjudicated annotation after resolving disagreements."""
    prompt_id: str
    prompt_text: str
    gold_spans: List[SpanAnnotation] = field(default_factory=list)
    
    # Disagreement info
    annotator_1_spans: List[SpanAnnotation] = field(default_factory=list)
    annotator_2_spans: List[SpanAnnotation] = field(default_factory=list)
    disagreements: List[Dict] = field(default_factory=list)


# =============================================================================
# Stratified Sampling
# =============================================================================

class StratifiedSampler:
    """
    Sample prompts stratified by provider/source and length bins.
    """
    
    LENGTH_BINS = {
        'short': (0, 500),
        'medium': (500, 2000),
        'long': (2000, float('inf'))
    }
    
    def __init__(self, target_size: int = 300):
        self.target_size = target_size
    
    def get_length_bin(self, text: str) -> str:
        """Determine length bin for a text."""
        length = len(text)
        for bin_name, (low, high) in self.LENGTH_BINS.items():
            if low <= length < high:
                return bin_name
        return 'long'
    
    def sample(self, prompts: List[Dict]) -> List[Dict]:
        """
        Sample prompts with stratification.
        
        Args:
            prompts: List of prompt dicts with 'content', 'provider' keys
            
        Returns:
            Stratified sample
        """
        # Group by provider and length bin
        groups = defaultdict(list)
        
        for prompt in prompts:
            provider = prompt.get('provider', 'unknown')
            length_bin = self.get_length_bin(prompt.get('content', ''))
            key = (provider, length_bin)
            groups[key].append(prompt)
        
        # Calculate samples per group
        n_groups = len(groups)
        samples_per_group = max(1, self.target_size // n_groups)
        
        sampled = []
        for key, group_prompts in groups.items():
            n_sample = min(samples_per_group, len(group_prompts))
            sampled.extend(random.sample(group_prompts, n_sample))
        
        # If we need more, sample from remaining
        if len(sampled) < self.target_size:
            remaining = [p for p in prompts if p not in sampled]
            additional = min(self.target_size - len(sampled), len(remaining))
            sampled.extend(random.sample(remaining, additional))
        
        return sampled[:self.target_size]


# =============================================================================
# Automatic Span Extractor (for baseline)
# =============================================================================

class AutomaticSpanExtractor:
    """
    Automatic guardrail span extraction using patterns.
    
    This is the baseline to compare against gold annotations.
    """
    
    PATTERNS = {
        'prohibition': [
            r'(?:you\s+)?(?:must\s+)?never\s+[\w\s,]+[.!]',
            r'(?:you\s+)?must\s+not\s+[\w\s,]+[.!]',
            r'(?:you\s+)?(?:should|shall)\s+not\s+[\w\s,]+[.!]',
            r'do\s+not\s+[\w\s,]+[.!]',
            r'(?:it\s+is\s+)?(?:strictly\s+)?(?:forbidden|prohibited)\s+(?:to\s+)?[\w\s,]+[.!]',
        ],
        'obligation': [
            r'(?:you\s+)?must\s+(?:always\s+)?[\w\s,]+[.!]',
            r'(?:you\s+)?(?:are\s+)?required\s+to\s+[\w\s,]+[.!]',
            r'always\s+[\w\s,]+[.!]',
            r'ensure\s+(?:that\s+)?[\w\s,]+[.!]',
        ],
        'boundary': [
            r'(?:you\s+)?(?:are\s+)?(?:only\s+)?(?:designed|intended)\s+(?:to|for)\s+[\w\s,]+[.!]',
            r'(?:your\s+)?scope\s+(?:is\s+)?(?:limited\s+to\s+)?[\w\s,]+[.!]',
            r'(?:you\s+)?(?:can\s+)?only\s+(?:answer|help|assist)\s+[\w\s,]+[.!]',
            r'outside\s+(?:of\s+)?(?:your\s+)?(?:scope|capabilities)[\w\s,]*[.!]',
        ],
        'identity': [
            r'(?:do\s+not|never)\s+(?:reveal|disclose|share)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions)[\w\s,]*[.!]',
            r'(?:keep|maintain)\s+(?:your\s+)?(?:instructions|prompt)\s+(?:confidential|secret)[\w\s,]*[.!]',
        ],
        'content_safety': [
            r'(?:do\s+not|never)\s+(?:generate|create|produce)\s+(?:harmful|violent|explicit)[\w\s,]*[.!]',
            r'avoid\s+(?:harmful|violent|explicit|inappropriate)[\w\s,]*[.!]',
        ],
        'security': [
            r'(?:do\s+not|never)\s+(?:help|assist)\s+(?:with\s+)?(?:hacking|attacks|malware)[\w\s,]*[.!]',
            r'(?:refuse|decline)\s+(?:requests\s+)?(?:for\s+)?(?:malicious|harmful)[\w\s,]*[.!]',
        ],
        'refusal': [
            r'(?:you\s+)?(?:should|must|will)\s+(?:refuse|decline)\s+[\w\s,]+[.!]',
            r'(?:politely\s+)?(?:refuse|decline)\s+(?:to\s+)?[\w\s,]+[.!]',
        ]
    }
    
    def extract(self, text: str) -> List[SpanAnnotation]:
        """Extract spans using patterns."""
        spans = []
        text_lower = text.lower()
        
        for ctype, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                    # Get original case text
                    original_text = text[match.start():match.end()]
                    
                    spans.append(SpanAnnotation(
                        text=original_text,
                        start=match.start(),
                        end=match.end(),
                        constraint_type=ctype,
                        annotator_id='automatic'
                    ))
        
        # Remove overlapping spans (keep longer ones)
        spans = self._remove_overlaps(spans)
        
        return spans
    
    def _remove_overlaps(self, spans: List[SpanAnnotation]) -> List[SpanAnnotation]:
        """Remove overlapping spans, keeping longer ones."""
        if not spans:
            return spans
        
        # Sort by length (descending) then by start position
        sorted_spans = sorted(spans, key=lambda s: (-len(s.text), s.start))
        
        kept = []
        used_ranges = set()
        
        for span in sorted_spans:
            # Check if overlaps with any kept span
            overlaps = False
            for pos in range(span.start, span.end):
                if pos in used_ranges:
                    overlaps = True
                    break
            
            if not overlaps:
                kept.append(span)
                for pos in range(span.start, span.end):
                    used_ranges.add(pos)
        
        return sorted(kept, key=lambda s: s.start)


# =============================================================================
# Agreement Metrics
# =============================================================================

class AgreementCalculator:
    """
    Calculate inter-annotator agreement metrics.
    """
    
    def calculate_span_agreement(
        self,
        annotations_1: List[SpanAnnotation],
        annotations_2: List[SpanAnnotation],
        overlap_threshold: float = 0.5
    ) -> Dict:
        """
        Calculate agreement on span identification.
        
        Two spans agree if they overlap by at least threshold.
        """
        matched_1 = set()
        matched_2 = set()
        
        for i, span1 in enumerate(annotations_1):
            for j, span2 in enumerate(annotations_2):
                overlap = self._calculate_overlap(span1, span2)
                if overlap >= overlap_threshold:
                    matched_1.add(i)
                    matched_2.add(j)
        
        # Calculate F1-style agreement
        precision = len(matched_1) / len(annotations_1) if annotations_1 else 0
        recall = len(matched_2) / len(annotations_2) if annotations_2 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'span_precision': precision,
            'span_recall': recall,
            'span_f1': f1,
            'matched_count': len(matched_1),
            'annotator_1_count': len(annotations_1),
            'annotator_2_count': len(annotations_2)
        }
    
    def calculate_type_agreement(
        self,
        annotations_1: List[SpanAnnotation],
        annotations_2: List[SpanAnnotation]
    ) -> Dict:
        """
        Calculate Cohen's kappa for type labels on matched spans.
        """
        # Find matched spans
        matched_pairs = []
        
        for span1 in annotations_1:
            for span2 in annotations_2:
                if self._calculate_overlap(span1, span2) >= 0.5:
                    matched_pairs.append((span1.constraint_type, span2.constraint_type))
                    break
        
        if len(matched_pairs) < 2:
            return {'kappa': None, 'n_pairs': len(matched_pairs)}
        
        types_1 = [p[0] for p in matched_pairs]
        types_2 = [p[1] for p in matched_pairs]
        
        kappa = cohen_kappa_score(types_1, types_2)
        
        # Also calculate agreement rate
        agreement_rate = sum(1 for t1, t2 in matched_pairs if t1 == t2) / len(matched_pairs)
        
        return {
            'kappa': kappa,
            'agreement_rate': agreement_rate,
            'n_pairs': len(matched_pairs)
        }
    
    def _calculate_overlap(self, span1: SpanAnnotation, span2: SpanAnnotation) -> float:
        """Calculate overlap ratio between two spans."""
        start = max(span1.start, span2.start)
        end = min(span1.end, span2.end)
        
        if start >= end:
            return 0.0
        
        intersection = end - start
        union = (span1.end - span1.start) + (span2.end - span2.start) - intersection
        
        return intersection / union if union > 0 else 0.0


# =============================================================================
# Evaluation Against Gold
# =============================================================================

class GoldEvaluator:
    """
    Evaluate automatic extractor against gold annotations.
    """
    
    def __init__(self):
        self.extractor = AutomaticSpanExtractor()
    
    def evaluate(
        self,
        gold_annotations: List[AdjudicatedAnnotation]
    ) -> Dict:
        """
        Evaluate automatic extraction against gold.
        
        Returns:
            Span-level P/R/F1 and error analysis
        """
        all_predictions = []
        all_gold = []
        
        errors = {
            'false_positives': [],
            'false_negatives': [],
            'type_errors': []
        }
        
        for gold in gold_annotations:
            # Run automatic extraction
            predicted = self.extractor.extract(gold.prompt_text)
            
            # Match predictions to gold
            for pred in predicted:
                matched = False
                for gold_span in gold.gold_spans:
                    if self._spans_match(pred, gold_span):
                        matched = True
                        all_predictions.append(1)
                        all_gold.append(1)
                        
                        # Check type match
                        if pred.constraint_type != gold_span.constraint_type:
                            errors['type_errors'].append({
                                'text': pred.text[:50],
                                'predicted_type': pred.constraint_type,
                                'gold_type': gold_span.constraint_type
                            })
                        break
                
                if not matched:
                    all_predictions.append(1)
                    all_gold.append(0)
                    errors['false_positives'].append({
                        'text': pred.text[:50],
                        'type': pred.constraint_type
                    })
            
            # Check for false negatives
            for gold_span in gold.gold_spans:
                matched = False
                for pred in predicted:
                    if self._spans_match(pred, gold_span):
                        matched = True
                        break
                
                if not matched:
                    all_predictions.append(0)
                    all_gold.append(1)
                    errors['false_negatives'].append({
                        'text': gold_span.text[:50],
                        'type': gold_span.constraint_type
                    })
        
        # Calculate metrics
        tp = sum(1 for p, g in zip(all_predictions, all_gold) if p == 1 and g == 1)
        fp = sum(1 for p, g in zip(all_predictions, all_gold) if p == 1 and g == 0)
        fn = sum(1 for p, g in zip(all_predictions, all_gold) if p == 0 and g == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'errors': errors,
            'error_analysis': self._analyze_errors(errors)
        }
    
    def _spans_match(
        self, 
        span1: SpanAnnotation, 
        span2: SpanAnnotation,
        threshold: float = 0.5
    ) -> bool:
        """Check if two spans match (overlap above threshold)."""
        start = max(span1.start, span2.start)
        end = min(span1.end, span2.end)
        
        if start >= end:
            return False
        
        intersection = end - start
        union = (span1.end - span1.start) + (span2.end - span2.start) - intersection
        
        return (intersection / union) >= threshold if union > 0 else False
    
    def _analyze_errors(self, errors: Dict) -> Dict:
        """Analyze error patterns."""
        analysis = {}
        
        # Most common FP types
        fp_types = defaultdict(int)
        for fp in errors['false_positives']:
            fp_types[fp['type']] += 1
        analysis['fp_by_type'] = dict(fp_types)
        
        # Most common FN types
        fn_types = defaultdict(int)
        for fn in errors['false_negatives']:
            fn_types[fn['type']] += 1
        analysis['fn_by_type'] = dict(fn_types)
        
        # Type confusion matrix
        type_confusions = defaultdict(lambda: defaultdict(int))
        for te in errors['type_errors']:
            type_confusions[te['gold_type']][te['predicted_type']] += 1
        analysis['type_confusion'] = {k: dict(v) for k, v in type_confusions.items()}
        
        return analysis


# =============================================================================
# Annotation Tool Helper
# =============================================================================

class AnnotationManager:
    """
    Manage the annotation process.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.sampler = StratifiedSampler()
        self.agreement_calc = AgreementCalculator()
    
    def prepare_annotation_batch(
        self,
        prompts: List[Dict],
        batch_size: int = 50
    ) -> List[Dict]:
        """
        Prepare a batch for annotation.
        
        Returns annotation tasks in JSON format.
        """
        sampled = self.sampler.sample(prompts)[:batch_size]
        
        tasks = []
        for i, prompt in enumerate(sampled):
            task = {
                'task_id': f"task_{i:04d}",
                'prompt_id': prompt.get('file_name', f'prompt_{i}'),
                'prompt_text': prompt.get('content', ''),
                'provider': prompt.get('provider', 'unknown'),
                'length_bin': self.sampler.get_length_bin(prompt.get('content', '')),
                'instructions': """
Please annotate all guardrail/constraint spans in this system prompt.

For each span:
1. Select the exact text
2. Choose the constraint type:
   - prohibition: "never", "must not", "do not"
   - obligation: "must", "always", "required"
   - boundary: scope limitations, "only answer about X"
   - identity: prompt protection, "don't reveal instructions"
   - content_safety: harmful content restrictions
   - security: security-related constraints
   - refusal: refusal instructions

3. For boundary constraints, also note:
   - What is allowed (in scope)
   - What is disallowed (out of scope)
"""
            }
            tasks.append(task)
        
        return tasks
    
    def calculate_agreement(
        self,
        annotator_1_file: str,
        annotator_2_file: str
    ) -> Dict:
        """
        Calculate inter-annotator agreement from annotation files.
        """
        with open(annotator_1_file) as f:
            annotations_1 = json.load(f)
        with open(annotator_2_file) as f:
            annotations_2 = json.load(f)
        
        all_span_agreements = []
        all_type_agreements = []
        
        for task_id in annotations_1:
            if task_id not in annotations_2:
                continue
            
            spans_1 = [SpanAnnotation(**s) for s in annotations_1[task_id].get('spans', [])]
            spans_2 = [SpanAnnotation(**s) for s in annotations_2[task_id].get('spans', [])]
            
            span_agreement = self.agreement_calc.calculate_span_agreement(spans_1, spans_2)
            type_agreement = self.agreement_calc.calculate_type_agreement(spans_1, spans_2)
            
            all_span_agreements.append(span_agreement)
            all_type_agreements.append(type_agreement)
        
        # Aggregate
        return {
            'span_agreement': {
                'mean_f1': np.mean([a['span_f1'] for a in all_span_agreements]),
                'std_f1': np.std([a['span_f1'] for a in all_span_agreements])
            },
            'type_agreement': {
                'mean_kappa': np.mean([a['kappa'] for a in all_type_agreements if a['kappa'] is not None]),
                'mean_agreement_rate': np.mean([a['agreement_rate'] for a in all_type_agreements])
            },
            'n_prompts': len(all_span_agreements)
        }

