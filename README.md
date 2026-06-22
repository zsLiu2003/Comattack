# COMA: When Compression Becomes an Attack Surface

Artifact for the paper **"When Compression Becomes an Attack Surface: Black-Box Attacks on Prompt-Compressed LLM Agents"**.

## Overview

COMA is a black-box adversarial framework that exploits prompt compression as an attack surface in LLM agent pipelines. It demonstrates that an adversary can craft inputs that, after compression, selectively remove or corrupt critical information -- causing downstream LLM agents to have misbehavior.

The framework implements a two-stage attack:
- **Stage I (Target Selection):** Identify which information to suppress (answer spans, guardrail negations, preference-critical keywords)
- **Stage II (Preimage Search):** Use COMA-based optimization to find adversarial inputs that, after compression, match the target

### Tasks
| Task | Abbrev. | Description |
|------|---------|-------------|
| Agent Tool Selection | ATS | Manipulate which tool/product the agent recommends |
| Question Answering | QA | Suppress answer spans so the agent cannot answer correctly |
| System Prompt Corruption | SPC | Remove guardrail negations (e.g., "do not" -> "") to disable safety rules |

### Compressors Evaluated
| Type | Compressor | Surrogate Model |
|------|-----------|-----------------|
| Extractive | LLMLingua-1 | Llama-2-7B |
| Extractive | LLMLingua-2 | xlm-roberta-large |
| Extractive | SelectiveContext | Llama-2-7B |
| Abstractive | Qwen3-4B | Qwen3-4B |
| Abstractive | Llama-3.2-3B | Llama-3.2-3B-Instruct |
| Abstractive | Gemma-3-4B | Gemma-3-4B |

## Quick Start

### 1. Install

```bash
conda create -n coma python=3.10 -y && conda activate coma
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
cd artifact/
pip install -e ".[all]"
```

## Reproducing Paper Results

Each RQ has a dedicated reproduction script:

| RQ | Script | Description |
|----|--------|-------------|
| RQ1 | `scripts/reproduce_rq1.sh` | Effectiveness: 3 tasks x 6 compressors
| RQ2 | `scripts/reproduce_rq2.sh` | Generalization: budget sweep + backend LLMs
| RQ3 | `scripts/reproduce_rq3.sh` | Surrogate mismatch + token retention
| RQ4 | `scripts/reproduce_rq4.sh` | Case studies: VSCode Cline, LangChain+Ollama
| RQ5 | `scripts/reproduce_rq5.sh` | Defense evaluation

## Output Format

Attack results are saved as JSONL files, one entry per line:

```json
{
    "context": "original input text...",
    "attacked_context": "adversarial input text...",
    "best_loss": 0.023,
    "converged": true,
    "steps": 142
}
```

## File Structure

```
artifact/
  README.md                 
  config.py                 # Configuration (env-var based, no hardcoded paths)
  requirements.txt          # Python dependencies
  .gitignore

  comattack/                # Core Python package
    __init__.py
    attacks/                # Stage II: preimage search (extractive + abstractive)
    compressors/            # Compressor wrappers
    defense/                # Defense baselines
    evaluation/             # End-to-end evaluation, metrics, compliance
    llm/                    # LLM provider abstraction (vLLM, OpenAI, Ollama)
    targets/                # Stage I: Target generation per task
    data/                   # Package data (prompts, templates)

  run_guardrail_attack.py   # Entry point: SPC task
  run_qa_attack.py          # Entry point: QA task
  run_pref_attack.py        # Entry point: ATS task
  run_surrogate_mismatch.py # Entry point: surrogate mismatch
  run_pref_attack.sh        # Batch launcher: ATS (all compressors)
  run_qa_attack.sh          # Batch launcher: QA (all compressors)
  run_surrogate_mismatch.sh  # Batch launcher: surrogate grid

```

