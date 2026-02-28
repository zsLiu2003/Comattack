"""
ICML Experiments Package
========================

Systematic evaluation of system prompt compression safety.

Modules:
- config: Central configuration and formal definitions
- metrics: Unified metric definitions (P0.2)
- baselines: Strong baselines (P0.5)
- gold_annotation: Gold annotation framework (P0.3)
- behavioral_eval: End-to-end behavioral evaluation (P0.4)
- mitigations: Mitigation experiments (P1.6)
- run_experiments: Main experiment runner
"""

from .config import (
    ExperimentConfig,
    ConstraintType,
    ConstraintSpan,
    DriftType,
    MetricConfig
)

from .metrics import (
    CompressionRatioCalculator,
    SpanPreservationCalculator,
    CatastrophicEditDetector
)

from .baselines import (
    BaselineManager,
    OriginalBaseline,
    TruncateFirstBaseline,
    RandomDeletionBaseline,
    StructurePreservingBaseline
)

__version__ = "1.0.0"
__all__ = [
    'ExperimentConfig',
    'ConstraintType', 
    'ConstraintSpan',
    'CompressionRatioCalculator',
    'SpanPreservationCalculator',
    'CatastrophicEditDetector',
    'BaselineManager'
]

