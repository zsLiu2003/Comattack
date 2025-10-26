# CompressionAttack

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/CompressionAttack.git
cd CompressionAttack
```

2. Install dependencies:
```bash
pip install torch transformers datasets flask llmlingua sentence-transformers bert-score python-Levenshtein
```

3. Configure model paths by copying and editing the configuration file:
```bash
cp config.py.example config.py
# Edit config.py with your model paths
```

## Configuration

Update `config.py` with your specific model paths:

```python
# Example configuration
COMPRESSION_MODEL_PATH = "NousResearch/Llama-2-7b-hf"  # or local path
LARGE_MODEL_PATH = "Qwen/Qwen3-7B"              # or local path
CUDA_DEVICE = "cuda:0"  # adjust based on your GPU setup
```

## Usage

### Basic Attack Example

```bash
# Run main attack pipeline
python src/main.py

# Run specific LLMLingua attack
python src/attacks/attack_llmlingua.py

# Try compression example
python examples/try_llmlingua.py
```

### Defense Evaluation

```bash
# Perplexity-based detection
python src/defense/ppl_detection.py

# LLM-based detection
python src/defense/LLM_detection.py
```

### Stealthiness Evaluation

Evaluate attack stealthiness using BERT-Score and semantic similarity:

```bash
# BERT-Score based stealthiness evaluation
python src/evaluators/stealthy_word.py

# Character-level edit distance stealthiness evaluation
python src/evaluators/stealthy_character.py
```

**Stealthiness Metrics:**
- **Composite Stealth Score**: `Stealth(C, C~) = λ · cosine_sim(C, C~) + (1 - λ) · BERTScore(C, C~)`
- **Semantic Similarity**: Using SentenceTransformers (all-mpnet-base-v2)
- **BERT-Score**: F1 score using bert-base-uncased with baseline rescaling
- **Character Similarity**: Normalized edit distance for character-level analysis

### Data Processing

```bash
# Generate datasets
python src/utils/data_generation.py

# Process existing data
python src/data/data_process.py
```

## Project Structure

```
CompressionAttack/
├── src/
│   ├── attacks/          # Attack implementations
│   │   ├── attack_llmlingua.py
│   │   ├── attack_llmlingua2.py
│   │   └── baseline_prompt.py
│   ├── defense/          # Defense mechanisms
│   │   ├── ppl_detection.py
│   │   ├── LLM_detection.py
│   │   └── bleu_defend.py
│   ├── utils/            # Core utilities
│   │   ├── get_ppl.py
│   │   ├── get_edit_token.py
│   │   ├── data_generation.py
│   │   └── main.py
│   ├── data/             # Datasets and data processing
│   │   ├── data_process.py
│   │   └── *.json        # Various datasets
│   └── evaluators/       # Evaluation scripts (stealthiness, BERT-Score, F1, etc.)
├── examples/             # Demo applications
│   ├── web/             # Flask web demos
│   ├── single_product_webpage/
│   └── try_llmlingua.py
└── config.py            # Configuration file
```