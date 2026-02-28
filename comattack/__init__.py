"""
Comattack -- Black-box adversarial testing via prompt compression.

Comattack (COMA) exploits prompt compression as an attack surface in LLM
agent pipelines. It implements a two-stage adversarial framework:

  Stage I  (targets/)  : Target selection -- identify what information to remove
  Stage II (attacks/)  : Preimage search -- find adversarial input via GCG

Supports three attack tasks:
  Task 1 : Preference Manipulation
  Task 2 : QA Information Suppression
  Task 3 : Guardrail Attenuation (System Prompt Corruption)

And two compressor families:
  HardCom : Extractive compressors (LLMLingua-1, LLMLingua-2)
  SoftCom : Abstractive compressors (Qwen3-4B, Llama-3.2-3B, Gemma-3-4B)
"""

__version__ = "0.1.0"
