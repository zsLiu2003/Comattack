"""
Stage II: Preimage Search (Attack) Module.

This module implements the core GCG-based adversarial attack algorithms:

- **HardCom** (extractive compressors like LLMLingua1/2):
  - Suffix optimization: ``hardcom_suffix.py``
  - Context editing: ``hardcom_context_edit.py``

- **SoftCom** (abstractive compressors / small LLMs):
  - Token replacement & suffix optimization: ``softcom.py``

- **Utilities**: ``gcg_utils.py`` (GCG primitives, AttackConfig)
- **Orchestration**: ``orchestration.py`` (run_gcg_attack, run_gcg_attack_smalllm)
- **External evaluation**: ``external_evaluator.py`` (ExternalEvaluator)
"""

from .gcg_utils import AttackConfig
from .hardcom_suffix import (
    AttackEvaluator,
    AttackforLLMLingua1,
    AttackforLLMLingua2,
    MultiplePromptsAttackforLLMlingua1,
    MultiplePromptsAttackforLLMlingua2,
)
from .hardcom_context_edit import (
    ContextEditAttackLLMLingua1,
    ContextEditAttackLLMLingua2,
    run_context_edit_attack,
)
from .softcom import (
    AttackforSmallLM,
    MultiplePromptsAttackforSmallLM,
)
from .external_evaluator import ExternalEvaluator, EvalResult
from .orchestration import run_gcg_attack, run_gcg_attack_smalllm

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
    "run_gcg_attack",
    "run_gcg_attack_smalllm",
]
