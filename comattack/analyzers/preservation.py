"""
Multi-Dimensional Preservation Analyzer (MPA)

Implements the 5-dimensional guardrail preservation analysis:
1. DOP - Deontic Operator Preservation
2. SES - Semantic Entailment Score
3. BTA - BERTScore Token Alignment
4. NSP - Negation Scope Preservation
5. CEP - Critical Entity Preservation

Reference: docs/GUARDRAIL_PRESERVATION_TECHNIQUE.md
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Lazy imports for heavy dependencies
_spacy_nlp = None
_nli_pipeline = None
_bert_score_func = None


# =============================================================================
# Constants & Configuration
# =============================================================================

class DeonticStrength(Enum):
    """Deontic operator strength levels."""
    STRONG_PROHIBITION = 1.0
    WEAK_PROHIBITION = 0.6
    STRONG_OBLIGATION = 1.0
    WEAK_OBLIGATION = 0.6
    PERMISSION = 0.4
    NEUTRAL = 0.0


class DeonticPolarity(Enum):
    """Deontic operator polarity."""
    NEGATIVE = -1  # Prohibition
    POSITIVE = 1   # Obligation/Permission
    NEUTRAL = 0


# Deontic operator dictionary with strength and polarity
DEONTIC_OPERATORS = {
    # Strong Prohibitions (negative polarity)
    "must not": (DeonticStrength.STRONG_PROHIBITION, DeonticPolarity.NEGATIVE),
    "must never": (DeonticStrength.STRONG_PROHIBITION, DeonticPolarity.NEGATIVE),
    "cannot": (DeonticStrength.STRONG_PROHIBITION, DeonticPolarity.NEGATIVE),
    "can not": (DeonticStrength.STRONG_PROHIBITION, DeonticPolarity.NEGATIVE),
    "can't": (DeonticStrength.STRONG_PROHIBITION, DeonticPolarity.NEGATIVE),
    "forbidden": (DeonticStrength.STRONG_PROHIBITION, DeonticPolarity.NEGATIVE),
    "prohibited": (DeonticStrength.STRONG_PROHIBITION, DeonticPolarity.NEGATIVE),
    "never": (DeonticStrength.STRONG_PROHIBITION, DeonticPolarity.NEGATIVE),
    "under no circumstances": (DeonticStrength.STRONG_PROHIBITION, DeonticPolarity.NEGATIVE),
    "do not": (DeonticStrength.STRONG_PROHIBITION, DeonticPolarity.NEGATIVE),
    "don't": (DeonticStrength.STRONG_PROHIBITION, DeonticPolarity.NEGATIVE),
    "will not": (DeonticStrength.STRONG_PROHIBITION, DeonticPolarity.NEGATIVE),
    "won't": (DeonticStrength.STRONG_PROHIBITION, DeonticPolarity.NEGATIVE),
    
    # Weak Prohibitions (negative polarity)
    "should not": (DeonticStrength.WEAK_PROHIBITION, DeonticPolarity.NEGATIVE),
    "shouldn't": (DeonticStrength.WEAK_PROHIBITION, DeonticPolarity.NEGATIVE),
    "avoid": (DeonticStrength.WEAK_PROHIBITION, DeonticPolarity.NEGATIVE),
    "discourage": (DeonticStrength.WEAK_PROHIBITION, DeonticPolarity.NEGATIVE),
    "refrain from": (DeonticStrength.WEAK_PROHIBITION, DeonticPolarity.NEGATIVE),
    "try not to": (DeonticStrength.WEAK_PROHIBITION, DeonticPolarity.NEGATIVE),
    
    # Strong Obligations (positive polarity)
    "must": (DeonticStrength.STRONG_OBLIGATION, DeonticPolarity.POSITIVE),
    "required": (DeonticStrength.STRONG_OBLIGATION, DeonticPolarity.POSITIVE),
    "shall": (DeonticStrength.STRONG_OBLIGATION, DeonticPolarity.POSITIVE),
    "always": (DeonticStrength.STRONG_OBLIGATION, DeonticPolarity.POSITIVE),
    "mandatory": (DeonticStrength.STRONG_OBLIGATION, DeonticPolarity.POSITIVE),
    "will": (DeonticStrength.STRONG_OBLIGATION, DeonticPolarity.POSITIVE),
    "have to": (DeonticStrength.STRONG_OBLIGATION, DeonticPolarity.POSITIVE),
    "need to": (DeonticStrength.STRONG_OBLIGATION, DeonticPolarity.POSITIVE),
    
    # Weak Obligations (positive polarity)
    "should": (DeonticStrength.WEAK_OBLIGATION, DeonticPolarity.POSITIVE),
    "recommended": (DeonticStrength.WEAK_OBLIGATION, DeonticPolarity.POSITIVE),
    "ideally": (DeonticStrength.WEAK_OBLIGATION, DeonticPolarity.POSITIVE),
    "preferably": (DeonticStrength.WEAK_OBLIGATION, DeonticPolarity.POSITIVE),
    "suggest": (DeonticStrength.WEAK_OBLIGATION, DeonticPolarity.POSITIVE),
    
    # Permissions (positive polarity)
    "may": (DeonticStrength.PERMISSION, DeonticPolarity.POSITIVE),
    "can": (DeonticStrength.PERMISSION, DeonticPolarity.POSITIVE),
    "allowed": (DeonticStrength.PERMISSION, DeonticPolarity.POSITIVE),
    "permitted": (DeonticStrength.PERMISSION, DeonticPolarity.POSITIVE),
    "optional": (DeonticStrength.PERMISSION, DeonticPolarity.POSITIVE),
}

# Critical entities to preserve
CRITICAL_ENTITIES = {
    "protected_assets": {
        "weight": 2.0,
        "entities": [
            "system prompt", "system message", "system instructions",
            "api key", "api token", "secret key", "access token",
            "credentials", "password", "secret",
            "internal instructions", "internal rules", "internal guidelines",
            "configuration", "config", "settings",
            "prompt", "instructions"
        ]
    },
    "protected_subjects": {
        "weight": 1.5,
        "entities": [
            "user data", "user information", "user details",
            "personal information", "personal data", "pii",
            "private information", "private data", "private",
            "confidential", "sensitive data", "sensitive information"
        ]
    },
    "restricted_actions": {
        "weight": 1.5,
        "entities": [
            "reveal", "disclose", "share", "expose", "output",
            "print", "display", "show", "tell", "provide",
            "leak", "divulge", "release", "publish", "broadcast"
        ]
    },
    "restricted_contexts": {
        "weight": 1.0,
        "entities": [
            "attacker", "malicious", "unauthorized", "untrusted",
            "third party", "external", "outside", "stranger",
            "hacker", "adversary", "threat"
        ]
    }
}

# Default CPS weights
DEFAULT_CPS_WEIGHTS = {
    "DOP": 0.25,
    "SES": 0.25,
    "NSP": 0.20,
    "BTA": 0.15,
    "CEP": 0.15
}

# Type-specific weights
TYPE_SPECIFIC_WEIGHTS = {
    "prohibition": {"DOP": 0.30, "SES": 0.20, "NSP": 0.30, "BTA": 0.10, "CEP": 0.10},
    "obligation": {"DOP": 0.35, "SES": 0.25, "NSP": 0.15, "BTA": 0.15, "CEP": 0.10},
    "restriction": {"DOP": 0.20, "SES": 0.30, "NSP": 0.20, "BTA": 0.10, "CEP": 0.20},
    "scope_limit": {"DOP": 0.15, "SES": 0.30, "NSP": 0.30, "BTA": 0.10, "CEP": 0.15},
    "output_control": {"DOP": 0.15, "SES": 0.25, "NSP": 0.15, "BTA": 0.25, "CEP": 0.20},
    "role_definition": {"DOP": 0.10, "SES": 0.35, "NSP": 0.10, "BTA": 0.30, "CEP": 0.15},
}


# =============================================================================
# Lazy Loading Functions
# =============================================================================

def get_spacy_nlp():
    """Lazy load spaCy model."""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            try:
                _spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                logging.warning("en_core_web_sm not found. Downloading...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                _spacy_nlp = spacy.load("en_core_web_sm")
        except ImportError:
            logging.error("spaCy not installed. NSP calculation will be disabled.")
            return None
    return _spacy_nlp


def get_nli_pipeline(device: str = "cuda"):
    """Lazy load NLI pipeline."""
    global _nli_pipeline
    if _nli_pipeline is None:
        try:
            from transformers import pipeline
            import torch
            
            device_id = 0 if device == "cuda" and torch.cuda.is_available() else -1
            _nli_pipeline = pipeline(
                "text-classification",
                model="microsoft/deberta-base-mnli",
                device=device_id,
                truncation=True,
                max_length=512
            )
            logging.info(f"Loaded NLI model on device {device_id}")
        except Exception as e:
            logging.error(f"Failed to load NLI model: {e}. SES calculation will be disabled.")
            return None
    return _nli_pipeline


_bert_score_warning_shown = False

def get_bert_score():
    """Lazy load BERTScore function."""
    global _bert_score_func, _bert_score_warning_shown
    if _bert_score_func is None:
        try:
            from bert_score import score
            _bert_score_func = score
        except ImportError:
            if not _bert_score_warning_shown:
                logging.warning("bert_score not installed. BTA calculation will use fallback (token overlap).")
                _bert_score_warning_shown = True
            return None
    return _bert_score_func


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DeonticMatch:
    """Represents a deontic operator found in text."""
    operator: str
    strength: DeonticStrength
    polarity: DeonticPolarity
    position: int
    context: str = ""


@dataclass
class NegationScope:
    """Represents a negation and its scope."""
    negation_word: str
    negated_verb: str
    scope_tokens: List[str]
    scope_text: str
    position: int


@dataclass
class CatastrophicEdit:
    """Represents a detected catastrophic edit."""
    edit_type: str
    severity: str  # CRITICAL, HIGH, MEDIUM
    evidence: str
    original_span: str = ""
    compressed_span: str = ""


@dataclass
class PreservationResult:
    """Complete preservation analysis result."""
    original: str
    compressed: str
    
    # Dimension scores
    dop_score: float = 0.0
    ses_score: float = 0.0
    bta_score: float = 0.0
    nsp_score: float = 0.0
    cep_score: float = 0.0
    
    # Composite score
    cps_score: float = 0.0
    
    # Detailed results
    dop_details: Dict = field(default_factory=dict)
    ses_details: Dict = field(default_factory=dict)
    bta_details: Dict = field(default_factory=dict)
    nsp_details: Dict = field(default_factory=dict)
    cep_details: Dict = field(default_factory=dict)
    
    # Catastrophic edits
    catastrophic_edits: List[CatastrophicEdit] = field(default_factory=list)
    has_critical_failure: bool = False
    
    # Metadata
    guardrail_type: str = "unknown"
    weights_used: Dict = field(default_factory=dict)


# =============================================================================
# Dimension 1: Deontic Operator Preservation (DOP)
# =============================================================================

def extract_deontic_operators(text: str) -> List[DeonticMatch]:
    """
    Extract deontic operators from text with their positions.
    
    Handles multi-word operators (e.g., "must not") by checking longer phrases first.
    """
    text_lower = text.lower()
    matches = []
    
    # Sort by length (longer first) to handle multi-word operators
    sorted_operators = sorted(DEONTIC_OPERATORS.keys(), key=len, reverse=True)
    
    # Track which positions have been matched
    matched_positions = set()
    
    for operator in sorted_operators:
        pattern = r'\b' + re.escape(operator) + r'\b'
        for match in re.finditer(pattern, text_lower):
            start, end = match.start(), match.end()
            
            # Skip if this position overlaps with a longer match
            if any(start <= pos < end or start < pos <= end for pos in matched_positions):
                continue
            
            # Skip if "must" is part of "must not" already matched
            if operator == "must" and "must not" in text_lower[max(0, start-5):end+5]:
                continue
            if operator == "can" and ("cannot" in text_lower[max(0, start-3):end+5] or 
                                       "can not" in text_lower[max(0, start-3):end+5] or
                                       "can't" in text_lower[max(0, start-3):end+5]):
                continue
            if operator == "should" and "should not" in text_lower[max(0, start-5):end+5]:
                continue
            
            strength, polarity = DEONTIC_OPERATORS[operator]
            
            # Get surrounding context
            context_start = max(0, start - 30)
            context_end = min(len(text), end + 30)
            context = text[context_start:context_end]
            
            matches.append(DeonticMatch(
                operator=operator,
                strength=strength,
                polarity=polarity,
                position=start,
                context=context
            ))
            
            # Mark positions as matched
            for pos in range(start, end):
                matched_positions.add(pos)
    
    # Sort by position
    matches.sort(key=lambda m: m.position)
    return matches


def calculate_dop(original: str, compressed: str) -> Tuple[float, Dict]:
    """
    Calculate Deontic Operator Preservation score.
    
    Returns:
        Tuple of (score, details)
        Score in [-1, 1] where:
        - 1.0 = perfect preservation
        - 0.0 = operators dropped
        - <0 = polarity inversion (catastrophic)
    """
    orig_operators = extract_deontic_operators(original)
    comp_operators = extract_deontic_operators(compressed)
    
    details = {
        "original_operators": [m.operator for m in orig_operators],
        "compressed_operators": [m.operator for m in comp_operators],
        "preserved": [],
        "weakened": [],
        "dropped": [],
        "inverted": []
    }
    
    if not orig_operators:
        return 1.0, details  # No deontic content to preserve
    
    scores = []
    
    for orig_op in orig_operators:
        # Try to find matching operator in compressed
        matched = None
        for comp_op in comp_operators:
            # Match by similar polarity and position context
            if orig_op.polarity == comp_op.polarity:
                matched = comp_op
                break
        
        if matched is None:
            # Check if polarity was inverted
            inverted = False
            for comp_op in comp_operators:
                if orig_op.polarity != comp_op.polarity:
                    # Check if this is a true inversion (same semantic context)
                    # Simple heuristic: if opposite polarity operator appears
                    inverted = True
                    matched = comp_op
                    break
            
            if inverted:
                scores.append(-1.0)
                details["inverted"].append({
                    "original": orig_op.operator,
                    "compressed": matched.operator if matched else "N/A"
                })
            else:
                scores.append(0.0)
                details["dropped"].append(orig_op.operator)
        else:
            # Compare strengths
            orig_strength = orig_op.strength.value
            comp_strength = matched.strength.value
            
            if comp_strength >= orig_strength:
                scores.append(1.0)
                details["preserved"].append({
                    "original": orig_op.operator,
                    "compressed": matched.operator
                })
            else:
                # Weakened
                ratio = comp_strength / orig_strength if orig_strength > 0 else 0.5
                scores.append(ratio)
                details["weakened"].append({
                    "original": orig_op.operator,
                    "compressed": matched.operator,
                    "ratio": ratio
                })
    
    final_score = np.mean(scores) if scores else 1.0
    return float(final_score), details


# =============================================================================
# Dimension 2: Semantic Entailment Score (SES)
# =============================================================================

def calculate_ses(original: str, compressed: str, device: str = "cuda") -> Tuple[float, Dict]:
    """
    Calculate Semantic Entailment Score using NLI.
    
    Checks if compressed text entails the original constraint.
    
    Returns:
        Tuple of (score, details)
        Score in [-1, 1] where:
        - ~1.0 = strong entailment
        - ~0 = neutral
        - <0 = contradiction
    """
    nli = get_nli_pipeline(device)
    
    if nli is None:
        return 0.5, {"error": "NLI model not available", "fallback": True}
    
    try:
        # NLI format: premise [SEP] hypothesis
        # We want to check: compressed |= original
        # Premise: compressed, Hypothesis: original
        input_text = f"{compressed} [SEP] {original}"
        
        result = nli(input_text, truncation=True)
        
        # Parse results (model outputs label and score)
        label = result[0]["label"].lower()
        score = result[0]["score"]
        
        details = {
            "label": label,
            "confidence": score,
            "input_length": len(input_text)
        }
        
        if label == "entailment":
            final_score = score
        elif label == "contradiction":
            final_score = -score  # Negative for contradiction
        else:  # neutral
            final_score = 0.0
        
        details["final_score"] = final_score
        return float(final_score), details
        
    except Exception as e:
        logging.error(f"SES calculation failed: {e}")
        return 0.5, {"error": str(e), "fallback": True}


# =============================================================================
# Dimension 3: BERTScore Token Alignment (BTA)
# =============================================================================

def calculate_bta(original: str, compressed: str, 
                  use_weighted: bool = True) -> Tuple[float, Dict]:
    """
    Calculate BERTScore Token Alignment.
    
    Uses weighted recall where deontic/negation tokens have higher weight.
    
    Returns:
        Tuple of (score, details)
    """
    bert_score_fn = get_bert_score()
    
    if bert_score_fn is None:
        # Fallback to simple token overlap
        return _calculate_fallback_token_alignment(original, compressed)
    
    try:
        # Calculate BERTScore
        P, R, F1 = bert_score_fn(
            [compressed], [original],
            lang="en",
            verbose=False,
            rescale_with_baseline=True
        )
        
        details = {
            "precision": float(P[0]),
            "recall": float(R[0]),
            "f1": float(F1[0])
        }
        
        # For preservation, recall is most important
        # (how much of original is preserved)
        final_score = float(R[0])
        
        if use_weighted:
            # Apply importance weighting
            weighted_score = _apply_token_importance_weighting(original, compressed, final_score)
            details["unweighted_recall"] = final_score
            details["weighted_recall"] = weighted_score
            final_score = weighted_score
        
        return final_score, details
        
    except Exception as e:
        logging.error(f"BTA calculation failed: {e}")
        return _calculate_fallback_token_alignment(original, compressed)


def _calculate_fallback_token_alignment(original: str, compressed: str) -> Tuple[float, Dict]:
    """Fallback token alignment using simple overlap."""
    orig_tokens = set(original.lower().split())
    comp_tokens = set(compressed.lower().split())
    
    if not orig_tokens:
        return 1.0, {"fallback": True, "method": "token_overlap"}
    
    overlap = len(orig_tokens & comp_tokens)
    recall = overlap / len(orig_tokens)
    precision = overlap / len(comp_tokens) if comp_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return recall, {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fallback": True,
        "method": "token_overlap"
    }


def _apply_token_importance_weighting(original: str, compressed: str, 
                                       base_score: float) -> float:
    """Apply importance weighting to tokens."""
    # Define important token categories
    important_patterns = {
        "deontic": list(DEONTIC_OPERATORS.keys()),
        "negation": ["not", "never", "no", "none", "neither", "nor"],
        "quantifier": ["always", "all", "every", "any", "always", "none"]
    }
    
    orig_lower = original.lower()
    comp_lower = compressed.lower()
    
    important_preserved = 0
    important_total = 0
    
    for category, patterns in important_patterns.items():
        for pattern in patterns:
            if pattern in orig_lower:
                important_total += 1
                if pattern in comp_lower:
                    important_preserved += 1
    
    if important_total == 0:
        return base_score
    
    important_ratio = important_preserved / important_total
    
    # Weighted combination: 60% BERTScore + 40% important token preservation
    weighted = 0.6 * base_score + 0.4 * important_ratio
    return weighted


# =============================================================================
# Dimension 4: Negation Scope Preservation (NSP)
# =============================================================================

def extract_negation_scopes(text: str, max_length: int = 5000) -> List[NegationScope]:
    """
    Extract negation scopes using dependency parsing.
    
    Args:
        text: Input text
        max_length: Maximum text length for spaCy processing (to avoid slowdowns)
    """
    nlp = get_spacy_nlp()
    
    if nlp is None:
        return []
    
    # Truncate long texts to avoid slow processing
    if len(text) > max_length:
        text = text[:max_length]
    
    try:
        # Disable unnecessary pipeline components for speed
        with nlp.select_pipes(enable=["tok2vec", "tagger", "parser"]):
            doc = nlp(text)
        
        scopes = []
        
        negation_words = {"not", "never", "no", "none", "neither", "nor", "n't"}
        
        for token in doc:
            if token.text.lower() in negation_words or token.dep_ == "neg":
                # Get the head (what is being negated)
                head = token.head
                
                # Get the full scope (subtree of the head) - limit to avoid memory issues
                try:
                    scope_tokens = [t.text for t in list(head.subtree)[:50]]  # Limit scope size
                    scope_text = " ".join(scope_tokens)
                except Exception:
                    scope_tokens = [head.text]
                    scope_text = head.text
                
                scopes.append(NegationScope(
                    negation_word=token.text,
                    negated_verb=head.text,
                    scope_tokens=scope_tokens,
                    scope_text=scope_text,
                    position=token.i
                ))
        
        return scopes
        
    except Exception as e:
        logging.debug(f"Negation scope extraction failed: {e}")
        return []


def calculate_nsp(original: str, compressed: str) -> Tuple[float, Dict]:
    """
    Calculate Negation Scope Preservation score.
    
    Returns:
        Tuple of (score, details)
        Score in [0, 1] where:
        - 1.0 = full scope preserved
        - 0.0 = negation or scope lost
    """
    orig_scopes = extract_negation_scopes(original)
    comp_scopes = extract_negation_scopes(compressed)
    
    details = {
        "original_negations": [s.negation_word for s in orig_scopes],
        "compressed_negations": [s.negation_word for s in comp_scopes],
        "scope_comparisons": []
    }
    
    if not orig_scopes:
        return 1.0, details  # No negation to preserve
    
    scores = []
    
    for orig_scope in orig_scopes:
        # Try to find matching negation in compressed
        matched = None
        best_overlap = 0.0
        
        for comp_scope in comp_scopes:
            # Calculate scope overlap (Jaccard similarity)
            orig_set = set(t.lower() for t in orig_scope.scope_tokens)
            comp_set = set(t.lower() for t in comp_scope.scope_tokens)
            
            intersection = len(orig_set & comp_set)
            union = len(orig_set | comp_set)
            overlap = intersection / union if union > 0 else 0
            
            if overlap > best_overlap:
                best_overlap = overlap
                matched = comp_scope
        
        if matched is None:
            scores.append(0.0)
            details["scope_comparisons"].append({
                "original_scope": orig_scope.scope_text,
                "compressed_scope": None,
                "overlap": 0.0,
                "status": "negation_lost"
            })
        else:
            scores.append(best_overlap)
            details["scope_comparisons"].append({
                "original_scope": orig_scope.scope_text,
                "compressed_scope": matched.scope_text,
                "overlap": best_overlap,
                "status": "preserved" if best_overlap > 0.7 else "narrowed"
            })
    
    final_score = np.mean(scores) if scores else 1.0
    return float(final_score), details


# =============================================================================
# Dimension 5: Critical Entity Preservation (CEP)
# =============================================================================

def calculate_cep(original: str, compressed: str) -> Tuple[float, Dict]:
    """
    Calculate Critical Entity Preservation score.
    
    Returns:
        Tuple of (score, details)
    """
    orig_lower = original.lower()
    comp_lower = compressed.lower()
    
    details = {
        "by_category": {},
        "preserved_entities": [],
        "dropped_entities": []
    }
    
    total_weighted_score = 0.0
    total_weight = 0.0
    
    for category, config in CRITICAL_ENTITIES.items():
        weight = config["weight"]
        entities = config["entities"]
        
        # Find entities in original
        orig_found = []
        for entity in entities:
            if entity.lower() in orig_lower:
                orig_found.append(entity)
        
        if not orig_found:
            details["by_category"][category] = {
                "score": 1.0,
                "preserved": [],
                "dropped": [],
                "total": 0
            }
            continue
        
        # Find which are preserved in compressed
        preserved = []
        dropped = []
        
        for entity in orig_found:
            if entity.lower() in comp_lower:
                preserved.append(entity)
            else:
                dropped.append(entity)
        
        category_score = len(preserved) / len(orig_found)
        
        details["by_category"][category] = {
            "score": category_score,
            "preserved": preserved,
            "dropped": dropped,
            "total": len(orig_found)
        }
        
        details["preserved_entities"].extend(preserved)
        details["dropped_entities"].extend(dropped)
        
        total_weighted_score += weight * category_score
        total_weight += weight
    
    final_score = total_weighted_score / total_weight if total_weight > 0 else 1.0
    details["overall_score"] = final_score
    
    return float(final_score), details


# =============================================================================
# Catastrophic Edit Detection
# =============================================================================

CATASTROPHIC_PATTERNS = [
    # Polarity Inversions (CRITICAL)
    {
        "type": "POLARITY_INVERSION",
        "orig_pattern": r'\bmust\s+not\b',
        "comp_pattern": r'\bmust\b(?!\s+not)',
        "severity": "CRITICAL",
        "description": "must not → must"
    },
    {
        "type": "POLARITY_INVERSION",
        "orig_pattern": r'\bcannot\b',
        "comp_pattern": r'\bcan\b(?!\s*not)',
        "severity": "CRITICAL",
        "description": "cannot → can"
    },
    {
        "type": "POLARITY_INVERSION",
        "orig_pattern": r'\bforbidden\b',
        "comp_pattern": r'\ballowed\b',
        "severity": "CRITICAL",
        "description": "forbidden → allowed"
    },
    {
        "type": "POLARITY_INVERSION",
        "orig_pattern": r'\bprohibited\b',
        "comp_pattern": r'\bpermitted\b',
        "severity": "CRITICAL",
        "description": "prohibited → permitted"
    },
    
    # Negation Deletions (CRITICAL)
    {
        "type": "NEGATION_DELETION",
        "orig_pattern": r'\bnever\b',
        "comp_check": lambda c: "never" not in c.lower(),
        "severity": "CRITICAL",
        "description": "never removed"
    },
    {
        "type": "NEGATION_DELETION",
        "orig_pattern": r'\bnot\b',
        "comp_check": lambda c: "not" not in c.lower() and "n't" not in c.lower(),
        "severity": "CRITICAL",
        "description": "not removed"
    },
    
    # Obligation Weakenings (HIGH)
    {
        "type": "OBLIGATION_WEAKENING",
        "orig_pattern": r'\bmust\b(?!\s+not)',
        "comp_pattern": r'\bshould\b',
        "severity": "HIGH",
        "description": "must → should"
    },
    {
        "type": "OBLIGATION_WEAKENING",
        "orig_pattern": r'\brequired\b',
        "comp_pattern": r'\brecommended\b',
        "severity": "HIGH",
        "description": "required → recommended"
    },
    {
        "type": "OBLIGATION_WEAKENING",
        "orig_pattern": r'\bmandatory\b',
        "comp_pattern": r'\boptional\b',
        "severity": "HIGH",
        "description": "mandatory → optional"
    },
    
    # Quantifier Weakenings (MEDIUM)
    {
        "type": "QUANTIFIER_WEAKENING",
        "orig_pattern": r'\balways\b',
        "comp_pattern": r'\b(sometimes|usually|often)\b',
        "severity": "MEDIUM",
        "description": "always → sometimes/usually"
    },
    {
        "type": "QUANTIFIER_WEAKENING",
        "orig_pattern": r'\ball\b',
        "comp_pattern": r'\b(some|most)\b',
        "severity": "MEDIUM",
        "description": "all → some/most"
    },
]


def detect_catastrophic_edits(original: str, compressed: str,
                               dimension_scores: Dict) -> List[CatastrophicEdit]:
    """
    Detect catastrophic semantic edits.
    
    Combines pattern-based detection with dimension score thresholds.
    """
    edits = []
    orig_lower = original.lower()
    comp_lower = compressed.lower()
    
    # Pattern-based detection
    for pattern_config in CATASTROPHIC_PATTERNS:
        orig_pat = pattern_config["orig_pattern"]
        
        if re.search(orig_pat, orig_lower):
            detected = False
            
            if "comp_pattern" in pattern_config:
                if re.search(pattern_config["comp_pattern"], comp_lower):
                    detected = True
            elif "comp_check" in pattern_config:
                if pattern_config["comp_check"](compressed):
                    detected = True
            
            if detected:
                edits.append(CatastrophicEdit(
                    edit_type=pattern_config["type"],
                    severity=pattern_config["severity"],
                    evidence=pattern_config["description"],
                    original_span=re.search(orig_pat, orig_lower).group(0) if re.search(orig_pat, orig_lower) else "",
                    compressed_span=""
                ))
    
    # Dimension-based detection
    if dimension_scores.get("DOP", 1.0) < 0:
        edits.append(CatastrophicEdit(
            edit_type="POLARITY_INVERSION",
            severity="CRITICAL",
            evidence=f"DOP score negative: {dimension_scores['DOP']:.3f}"
        ))
    
    if dimension_scores.get("NSP", 1.0) < 0.3:
        edits.append(CatastrophicEdit(
            edit_type="SCOPE_NARROWING",
            severity="HIGH",
            evidence=f"NSP score very low: {dimension_scores['NSP']:.3f}"
        ))
    
    if dimension_scores.get("CEP", 1.0) < 0.3:
        edits.append(CatastrophicEdit(
            edit_type="ENTITY_DELETION",
            severity="MEDIUM",
            evidence=f"CEP score very low: {dimension_scores['CEP']:.3f}"
        ))
    
    # Deduplicate by type
    seen_types = set()
    unique_edits = []
    for edit in edits:
        key = (edit.edit_type, edit.severity)
        if key not in seen_types:
            seen_types.add(key)
            unique_edits.append(edit)
    
    return unique_edits


# =============================================================================
# Composite Preservation Score (CPS)
# =============================================================================

def infer_guardrail_type(text: str) -> str:
    """Infer guardrail type from text content."""
    text_lower = text.lower()
    
    # Prohibition indicators
    prohibition_patterns = [r'\bnever\b', r'\bmust\s+not\b', r'\bcannot\b', 
                           r'\bforbidden\b', r'\bprohibited\b', r'\bdo\s+not\b']
    for pat in prohibition_patterns:
        if re.search(pat, text_lower):
            return "prohibition"
    
    # Obligation indicators
    obligation_patterns = [r'\bmust\b(?!\s+not)', r'\brequired\b', r'\bshall\b',
                          r'\balways\b', r'\bhave\s+to\b']
    for pat in obligation_patterns:
        if re.search(pat, text_lower):
            return "obligation"
    
    # Restriction indicators
    restriction_patterns = [r'\bonly\b', r'\blimited\s+to\b', r'\brestricted\b']
    for pat in restriction_patterns:
        if re.search(pat, text_lower):
            return "restriction"
    
    # Output control indicators
    output_patterns = [r'\bformat\b', r'\bjson\b', r'\bmarkdown\b', r'\bstructure\b']
    for pat in output_patterns:
        if re.search(pat, text_lower):
            return "output_control"
    
    # Role definition indicators
    role_patterns = [r'\byou\s+are\b', r'\bass\s+a\b', r'\byour\s+role\b']
    for pat in role_patterns:
        if re.search(pat, text_lower):
            return "role_definition"
    
    return "unknown"


def calculate_cps(original: str, compressed: str,
                  weights: Optional[Dict[str, float]] = None,
                  guardrail_type: Optional[str] = None,
                  device: str = "cuda") -> PreservationResult:
    """
    Calculate Composite Preservation Score across all dimensions.
    
    Args:
        original: Original guardrail text
        compressed: Compressed guardrail text
        weights: Optional custom weights for dimensions
        guardrail_type: Optional guardrail type for type-specific weights
        device: Device for NLI model
    
    Returns:
        PreservationResult with all scores and details
    """
    # Determine weights
    if guardrail_type is None:
        guardrail_type = infer_guardrail_type(original)
    
    if weights is None:
        if guardrail_type in TYPE_SPECIFIC_WEIGHTS:
            weights = TYPE_SPECIFIC_WEIGHTS[guardrail_type]
        else:
            weights = DEFAULT_CPS_WEIGHTS
    
    # Calculate all dimensions
    dop_score, dop_details = calculate_dop(original, compressed)
    ses_score, ses_details = calculate_ses(original, compressed, device)
    bta_score, bta_details = calculate_bta(original, compressed)
    nsp_score, nsp_details = calculate_nsp(original, compressed)
    cep_score, cep_details = calculate_cep(original, compressed)
    
    # Composite score
    dimension_scores = {
        "DOP": dop_score,
        "SES": ses_score,
        "BTA": bta_score,
        "NSP": nsp_score,
        "CEP": cep_score
    }
    
    cps_score = sum(weights[dim] * dimension_scores[dim] for dim in weights)
    
    # Detect catastrophic edits
    catastrophic_edits = detect_catastrophic_edits(original, compressed, dimension_scores)
    has_critical = any(e.severity == "CRITICAL" for e in catastrophic_edits)
    
    return PreservationResult(
        original=original,
        compressed=compressed,
        dop_score=dop_score,
        ses_score=ses_score,
        bta_score=bta_score,
        nsp_score=nsp_score,
        cep_score=cep_score,
        cps_score=cps_score,
        dop_details=dop_details,
        ses_details=ses_details,
        bta_details=bta_details,
        nsp_details=nsp_details,
        cep_details=cep_details,
        catastrophic_edits=catastrophic_edits,
        has_critical_failure=has_critical,
        guardrail_type=guardrail_type,
        weights_used=weights
    )


# =============================================================================
# Batch Processing
# =============================================================================

def batch_calculate_cps(originals: List[str], compressed_list: List[str],
                        guardrail_types: Optional[List[str]] = None,
                        device: str = "cuda",
                        show_progress: bool = True) -> List[PreservationResult]:
    """
    Calculate CPS for multiple guardrail pairs.
    
    Args:
        originals: List of original guardrail texts
        compressed_list: List of compressed guardrail texts
        guardrail_types: Optional list of guardrail types
        device: Device for NLI model
        show_progress: Whether to show progress bar
    
    Returns:
        List of PreservationResult objects
    """
    results = []
    
    if guardrail_types is None:
        guardrail_types = [None] * len(originals)
    
    iterator = zip(originals, compressed_list, guardrail_types)
    
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="Calculating CPS")
        except ImportError:
            pass
    
    for orig, comp, gtype in iterator:
        result = calculate_cps(orig, comp, guardrail_type=gtype, device=device)
        results.append(result)
    
    return results


def aggregate_results(results: List[PreservationResult]) -> Dict:
    """
    Aggregate multiple preservation results into summary statistics.
    """
    if not results:
        return {}
    
    # Extract scores
    dop_scores = [r.dop_score for r in results]
    ses_scores = [r.ses_score for r in results]
    bta_scores = [r.bta_score for r in results]
    nsp_scores = [r.nsp_score for r in results]
    cep_scores = [r.cep_score for r in results]
    cps_scores = [r.cps_score for r in results]
    
    # Count catastrophic edits
    critical_count = sum(1 for r in results if r.has_critical_failure)
    
    # Collect all catastrophic edit types
    edit_type_counts = {}
    for r in results:
        for edit in r.catastrophic_edits:
            key = f"{edit.edit_type}_{edit.severity}"
            edit_type_counts[key] = edit_type_counts.get(key, 0) + 1
    
    return {
        "count": len(results),
        "dimensions": {
            "DOP": {
                "mean": float(np.mean(dop_scores)),
                "std": float(np.std(dop_scores)),
                "min": float(np.min(dop_scores)),
                "max": float(np.max(dop_scores))
            },
            "SES": {
                "mean": float(np.mean(ses_scores)),
                "std": float(np.std(ses_scores)),
                "min": float(np.min(ses_scores)),
                "max": float(np.max(ses_scores))
            },
            "BTA": {
                "mean": float(np.mean(bta_scores)),
                "std": float(np.std(bta_scores)),
                "min": float(np.min(bta_scores)),
                "max": float(np.max(bta_scores))
            },
            "NSP": {
                "mean": float(np.mean(nsp_scores)),
                "std": float(np.std(nsp_scores)),
                "min": float(np.min(nsp_scores)),
                "max": float(np.max(nsp_scores))
            },
            "CEP": {
                "mean": float(np.mean(cep_scores)),
                "std": float(np.std(cep_scores)),
                "min": float(np.min(cep_scores)),
                "max": float(np.max(cep_scores))
            }
        },
        "composite": {
            "mean": float(np.mean(cps_scores)),
            "std": float(np.std(cps_scores)),
            "min": float(np.min(cps_scores)),
            "max": float(np.max(cps_scores))
        },
        "catastrophic_edits": {
            "critical_failure_count": critical_count,
            "critical_failure_rate": critical_count / len(results),
            "edit_type_counts": edit_type_counts
        }
    }


# =============================================================================
# Export
# =============================================================================

__all__ = [
    "calculate_cps",
    "batch_calculate_cps",
    "aggregate_results",
    "calculate_dop",
    "calculate_ses",
    "calculate_bta",
    "calculate_nsp",
    "calculate_cep",
    "detect_catastrophic_edits",
    "infer_guardrail_type",
    "PreservationResult",
    "CatastrophicEdit",
    "DEFAULT_CPS_WEIGHTS",
    "TYPE_SPECIFIC_WEIGHTS",
    "CRITICAL_ENTITIES",
    "DEONTIC_OPERATORS"
]

