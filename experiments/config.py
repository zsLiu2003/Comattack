"""
ICML Experiment Configuration
=============================

Central configuration for all experiments addressing P0 requirements.

NOTE: All paths are defined relative to PROJECT_ROOT.
      When moving the project folder, only need to change PROJECT_ROOT.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import os


# =============================================================================
# PROJECT ROOT - CHANGE THIS WHEN MOVING THE PROJECT
# =============================================================================

# Auto-detect project root (parent of experiments folder)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# Or manually set if needed:
# PROJECT_ROOT = "/path/to/your/SystemCom"


# =============================================================================
# PATH CONFIGURATION - All paths relative to PROJECT_ROOT
# =============================================================================

class Paths:
    """
    Centralized path management.
    
    Usage:
        from experiments.config import Paths
        
        data_path = Paths.CONSOLIDATED_DATA
        output_dir = Paths.COMPRESSION_RESULTS
    """
    
    # Data directories
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    CONSOLIDATED_DIR = os.path.join(DATA_DIR, "consolidated_guardrails")
    CONSOLIDATED_DATA = os.path.join(CONSOLIDATED_DIR, "consolidated_guardrails.json")
    
    # Output directories
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
    PPL_ANALYSIS = os.path.join(OUTPUT_DIR, "ppl_analysis")
    COMPRESSION_RESULTS = os.path.join(PROJECT_ROOT, "compression_results")
    COMPRESSION_HARD = os.path.join(COMPRESSION_RESULTS, "hard")
    COMPRESSION_SOFT = os.path.join(COMPRESSION_RESULTS, "soft")
    
    # Experiment outputs
    EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
    SOFT_ANALYSIS = os.path.join(EXPERIMENTS_DIR, "soft_analysis")
    ICML_RESULTS = os.path.join(PROJECT_ROOT, "icml_results")
    
    # Model cache
    MODEL_CACHE = os.path.join(PROJECT_ROOT, ".model_cache")
    
    # External dependencies
    AUTOCOMPRESSOR_DIR = os.path.join(PROJECT_ROOT, "AutoCompressors")
    ICAE_DIR = os.path.join(PROJECT_ROOT, "icae")
    
    @classmethod
    def ensure_dirs(cls):
        """Create all output directories if they don't exist."""
        dirs_to_create = [
            cls.OUTPUT_DIR,
            cls.PPL_ANALYSIS,
            cls.COMPRESSION_RESULTS,
            cls.COMPRESSION_HARD,
            cls.COMPRESSION_SOFT,
            cls.SOFT_ANALYSIS,
            cls.ICML_RESULTS,
        ]
        for d in dirs_to_create:
            os.makedirs(d, exist_ok=True)
    
    @classmethod
    def get_path(cls, name: str) -> str:
        """Get path by name string."""
        return getattr(cls, name.upper(), None)
    
    @classmethod
    def print_all(cls):
        """Print all configured paths for debugging."""
        print("=" * 60)
        print(f"PROJECT_ROOT: {PROJECT_ROOT}")
        print("=" * 60)
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith('_'):
                print(f"  {attr}: {getattr(cls, attr)}")
        print("=" * 60)

# =============================================================================
# P0.1: Constraint Drift Formal Definitions
# =============================================================================

class ConstraintType(Enum):
    """
    Guardrail/Constraint taxonomy (7 classes including Boundary).
    
    Definition: A constraint is a text span in the system prompt that 
    specifies a behavioral requirement or restriction for the LLM.
    """
    PROHIBITION = "prohibition"          # "NEVER", "must not", "do not"
    OBLIGATION = "obligation"            # "must", "always", "required"
    BOUNDARY = "boundary"                # "scope", "only answer about X"
    IDENTITY_PROTECTION = "identity"     # "do not reveal system prompt"
    CONTENT_SAFETY = "content_safety"    # "no harmful content"
    SECURITY = "security"                # "no hacking instructions"
    REFUSAL = "refusal"                  # "decline", "refuse"


class PreservationType(Enum):
    """
    How preservation is defined for different compressor types.
    
    Hard compression: Text span is literally retained (substring match)
    Soft compression: Semantic entailment (compressed output entails original constraint)
    """
    HARD_LITERAL = "hard_literal"        # Exact or fuzzy substring match
    SOFT_ENTAILMENT = "soft_entailment"  # NLI-based entailment check


class DriftType(Enum):
    """
    Types of constraint drift (P1.3: Catastrophic edits).
    """
    PRESERVED = "preserved"              # Constraint fully preserved
    DROPPED = "dropped"                  # Constraint completely removed
    NEGATION_FLIP = "negation_flip"      # "must not" → "must"
    OBLIGATION_WEAKENED = "weakened"     # "must" → "should/may"
    PARAPHRASED = "paraphrased"          # Reworded but semantically equivalent
    CONTRADICTED = "contradicted"        # Meaning reversed


# =============================================================================
# P0.2: Metric Definitions
# =============================================================================

@dataclass
class MetricConfig:
    """
    Unified metric definitions with explicit formulas.
    
    Compression Ratio (CR):
        CR = tokens(compressed) / tokens(original)
        Tokenizer: downstream model tokenizer (Qwen/LLaMA)
    
    Span Preservation Rate (macro):
        MacroPreserve = (1/N) * Σ_i [preserved_spans_i / total_spans_i]
        where i is each prompt
    
    Span Preservation Rate (micro):
        MicroPreserve = Σ_i preserved_spans_i / Σ_i total_spans_i
        Occurrence-weighted across all prompts
    """
    # Tokenizer for CR calculation
    tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # Span matching threshold for hard compression
    fuzzy_match_threshold: float = 0.85  # Levenshtein ratio
    
    # Entailment threshold for soft compression
    entailment_threshold: float = 0.7  # NLI probability
    
    # Report both macro and micro
    report_macro: bool = True
    report_micro: bool = True


# =============================================================================
# P0.5: Baseline Configuration
# =============================================================================

@dataclass
class BaselineConfig:
    """
    Strong baselines to rule out confounds.
    """
    baselines: List[str] = field(default_factory=lambda: [
        "original",              # No compression
        "truncate_first",        # Keep first N tokens
        "truncate_last",         # Keep last N tokens
        "random_deletion",       # Random token deletion to budget
        "structure_preserving",  # Keep headings/bullets, delete bodies
    ])


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """
    
    Uses Paths class for all path management.
    """
    # Project paths (use Paths class)
    project_root: str = field(default_factory=lambda: PROJECT_ROOT)
    data_dir: str = field(default_factory=lambda: Paths.CONSOLIDATED_DIR)
    output_dir: str = field(default_factory=lambda: Paths.ICML_RESULTS)
    
    # Compressors (P0.5)
    compressors: List[str] = field(default_factory=lambda: [
        "llmlingua",
        "llmlingua2", 
        "autocompressor"
    ])
    
    # Compression ratios
    compression_ratios: List[float] = field(default_factory=lambda: [
        0.4, 0.6, 0.8  # Main ratios
    ])
    stress_ratios: List[float] = field(default_factory=lambda: [0.2])  # Optional
    
    # Downstream models for behavioral evaluation (P0.4)
    downstream_models: List[str] = field(default_factory=lambda: [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct"
    ])
    
    # Gold annotation config (P0.3)
    gold_sample_size: int = 300  # 250-500 recommended
    num_annotators: int = 2
    
    # Behavioral evaluation config (P0.4)
    queries_per_prompt: int = 6  # 2 in-scope, 2 out-of-scope, 2 stress
    
    # Random seeds for reproducibility
    seed: int = 42
    
    # Device
    device: str = "cuda"
    
    def __post_init__(self):
        # Ensure directories exist
        Paths.ensure_dirs()


# =============================================================================
# Constraint Span Definition (P0.1)
# =============================================================================

@dataclass
class ConstraintSpan:
    """
    A constraint/guardrail span in a system prompt.
    
    Definition: A contiguous text unit that specifies a behavioral 
    requirement or restriction for the LLM.
    """
    text: str                           # The span text
    start: int                          # Start character position
    end: int                            # End character position
    constraint_type: ConstraintType     # One of 7 classes
    confidence: float = 1.0             # Annotation confidence
    
    # For Boundary constraints (P1.1)
    allowed_scope: Optional[str] = None   # What's in scope
    disallowed_scope: Optional[str] = None  # What's out of scope
    
    # Annotation metadata
    annotator_id: Optional[str] = None
    is_gold: bool = False               # Part of gold set


@dataclass 
class ConstraintDriftResult:
    """
    Result of constraint drift analysis for a single span.
    """
    original_span: ConstraintSpan
    drift_type: DriftType
    preserved_text: Optional[str] = None  # For hard compression
    entailment_score: Optional[float] = None  # For soft compression (P1.2)
    
    # Catastrophic edit detection (P1.3)
    is_negation_flip: bool = False
    is_obligation_weakened: bool = False


# =============================================================================
# Behavioral Evaluation Config (P0.4)
# =============================================================================

@dataclass
class BehavioralEvalConfig:
    """
    Configuration for end-to-end behavioral evaluation.
    """
    # Query types for each prompt
    query_types: List[str] = field(default_factory=lambda: [
        "in_scope",           # Should answer
        "out_of_scope",       # Should refuse (boundary)
        "confidentiality",    # Should not leak prompt
        "refusal_test",       # Should refuse harmful
        "compliance_test",    # Should comply with format
        "stress_test"         # Edge cases
    ])
    
    # Judging
    use_llm_judge: bool = True
    judge_model: str = "Qwen/Qwen2.5-72B-Instruct"  # Or LlamaGuard
    human_validation_sample: int = 100  # Sample for calibration
    
    # Metrics (P0.4)
    metrics: List[str] = field(default_factory=lambda: [
        "violation_rate",     # Constraint violated when shouldn't
        "over_refusal_rate",  # Refuses when should answer
        "delta_violation",    # compressed - original
        "leakage_rate",       # System prompt disclosed
    ])
    
    # Statistical significance
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95


# Create default config
DEFAULT_CONFIG = ExperimentConfig()
