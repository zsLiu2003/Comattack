# Configuration file for COMA artifact
# All paths are relative to the artifact root or configurable via environment variables.

import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "comattack", "data")

# Model paths -- models are downloaded automatically from HuggingFace.
# Set COMA_MODEL_CACHE to override the default HuggingFace cache location.
MODEL_BASE_PATH = os.environ.get("COMA_MODEL_CACHE", os.path.join(PROJECT_ROOT, "models"))

# Dataset paths (artifact data directory)
ARTIFACT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
QA_DATA_DIR = os.path.join(ARTIFACT_DATA_DIR, "qa")
ATS_DATA_DIR = os.path.join(ARTIFACT_DATA_DIR, "ats")
SPC_DATA_DIR = os.path.join(ARTIFACT_DATA_DIR, "system_prompt")

# Output paths
DEFAULT_OUTPUT_PATH = os.environ.get("COMA_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "results"))

# CUDA settings
DEFAULT_CUDA_DEVICE = os.environ.get("COMA_DEVICE", "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
