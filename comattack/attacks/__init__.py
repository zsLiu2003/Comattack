"""
Stage II: Preimage Search (Attack) Module.

This module implements the core COMA-based adversarial attack algorithms:

- **Extractive compressors** (LLMLingua1/2, SelectiveContext):
  - Suffix optimization: ``extractive_suffix.py``
  - Context editing: ``extractive_context_edit.py``

- **Summarization-based compressors** (small LLMs):
  - Token replacement & suffix optimization: ``summarize_based.py``

- **Utilities**: ``coma_utils.py`` (COMA primitives, AttackConfig)
- **Orchestration**: ``orchestration.py`` (run_coma_attack, run_coma_attack_smalllm)
- **External evaluation**: ``external_evaluator.py`` (ExternalEvaluator)
"""

from .coma_utils import AttackConfig
from .extractive_suffix import (
    AttackEvaluator,
    AttackforLLMLingua1,
    AttackforLLMLingua2,
    MultiplePromptsAttackforLLMlingua1,
    MultiplePromptsAttackforLLMlingua2,
)
from .extractive_context_edit import (
    ContextEditAttackLLMLingua1,
    ContextEditAttackLLMLingua2,
    run_context_edit_attack,
)
from .summarize_based import (
    AttackforSmallLM,
    MultiplePromptsAttackforSmallLM,
)
from .external_evaluator import ExternalEvaluator, EvalResult
from .orchestration import run_coma_attack, run_coma_attack_smalllm

__all__ = [
    "AttackConfig",
    "AttackEvaluator",
    "AttackforLLMLingua1",
    "AttackforLLMLingua2",
    "MultiplePromptsAttackforLLMlingua1",
    "MultiplePromptsAttackforLLMlingua2",
    "ContextEditAttackLLMLingua1",
    "ContextEditAttackLLMLingua2",
    "run_context_edit_attack",
    "AttackforSmallLM",
    "MultiplePromptsAttackforSmallLM",
    "ExternalEvaluator",
    "EvalResult",
    "run_coma_attack",
    "run_coma_attack_smalllm",
]
