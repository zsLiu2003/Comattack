# Configuration file for CompressionAttack project
# Modify these paths according to your environment

import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DATA_DIR = os.path.join(SRC_DIR, "data")

# Model paths - Update these to point to your model directories
MODEL_BASE_PATH = "models"  # Change this to your model directory
COMPRESSION_MODEL_PATH = os.path.join(MODEL_BASE_PATH, "gpt2-dolly")
LARGE_MODEL_PATH = os.path.join(MODEL_BASE_PATH, "Qwen3-32B")
GPT2_LARGE_PATH = os.path.join(MODEL_BASE_PATH, "gpt2-large")
LLAMA2_PATH = os.path.join(MODEL_BASE_PATH, "Llama-2-7b-chat-hf")
LLAMA3_PATH = os.path.join(MODEL_BASE_PATH, "Llama-3-8B-Instruct")
PHI4_PATH = os.path.join(MODEL_BASE_PATH, "phi-4")
MISTRAL_PATH = os.path.join(MODEL_BASE_PATH, "Mistral-7B-Instruct-v0.2")
SENTENCE_MODEL_PATH = os.path.join(MODEL_BASE_PATH, "all-mpnet-base-v2")

# Dataset paths
DEFAULT_DATASET = os.path.join(DATA_DIR, "data.json")
KEYWORDS_DATASET = os.path.join(DATA_DIR, "new_keywords_Qwen3.json")
TARGET_DEMO_DATASET = os.path.join(DATA_DIR, "data_best_Qwen3.json")
QA_DATASET = os.path.join(DATA_DIR, "QA_keywords_edit.json")

# Output paths
DEFAULT_OUTPUT_PATH = DATA_DIR

# CUDA settings
DEFAULT_CUDA_DEVICE = "cuda:0" if os.environ.get('CUDA_VISIBLE_DEVICES') else "cpu"