"""
Experiments Package
========================

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

