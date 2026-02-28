#!/bin/bash
# ── QA Attack (Task ②) ───────────────────────────────────────────────
# Run COMA attacks on the QA task for different compressors.
# Usage: bash run_qa_attack.sh

set -e
cd "$(dirname "$0")"

DATA_DIR="/hdd2/zesen/Comattack/src/data"
RESULT_DIR="results/qa"
BASE_MODEL="NousResearch/Llama-2-7b-hf"

# ── Shared parameters ────────────────────────────────────────────────
NUM_STEPS=200
BATCH_SIZE=256
TOPK=64
EVAL_BATCH=128
TEST_STEPS=10
SEED=42
MAX_ENTRIES=-1  # -1 = all

# =====================================================================
# HardCom: extractive compressors
# =====================================================================

# --- LLMLingua1 (surrogate: Llama-2) ---
echo "=== QA | HardCom | LLMLingua1 ==="
python run_qa_attack.py \
    --data "${DATA_DIR}/squad_qa_filtered_llmlingua1_max900.json" \
    --compressor llmlingua1 \
    --surrogate-model NousResearch/Llama-2-7b-hf \
    --num-steps ${NUM_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --topk ${TOPK} \
    --eval-batch-size ${EVAL_BATCH} \
    --test-steps ${TEST_STEPS} \
    --seed ${SEED} \
    --max-entries ${MAX_ENTRIES} \
    --output "${RESULT_DIR}/hardcom_llmlingua1"

# --- LLMLingua2 (surrogate: BERT-based classifier) ---
echo "=== QA | HardCom | LLMLingua2 ==="
python run_qa_attack.py \
    --data "${DATA_DIR}/squad_qa_filtered_llmlingua2_max460.json" \
    --compressor llmlingua2 \
    --surrogate-model microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank \
    --num-steps ${NUM_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --topk ${TOPK} \
    --eval-batch-size ${EVAL_BATCH} \
    --test-steps ${TEST_STEPS} \
    --seed ${SEED} \
    --max-entries ${MAX_ENTRIES} \
    --output "${RESULT_DIR}/hardcom_llmlingua2"

# =====================================================================
# SoftCom: abstractive compressors (small LLMs)
# =====================================================================
COMP_TOKENS=200

# --- Qwen3-4B ---
echo "=== QA | SoftCom | Qwen3-4B ==="
python run_qa_attack.py \
    --data "${DATA_DIR}/squad_qa_filtered_llmlingua1_max900.json" \
    --compressor qwen3-4b \
    --surrogate-model Qwen/Qwen3-4B \
    --num-steps ${NUM_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --topk ${TOPK} \
    --eval-batch-size ${EVAL_BATCH} \
    --test-steps ${TEST_STEPS} \
    --seed ${SEED} \
    --max-entries ${MAX_ENTRIES} \
    --compression-target-tokens ${COMP_TOKENS} \
    --output "${RESULT_DIR}/softcom_qwen3_4b"

# --- Llama-3.2-3B ---
echo "=== QA | SoftCom | Llama-3.2-3B ==="
python run_qa_attack.py \
    --data "${DATA_DIR}/squad_qa_filtered_llmlingua1_max900.json" \
    --compressor llama-3.2-3b \
    --surrogate-model meta-llama/Llama-3.2-3B-Instruct \
    --num-steps ${NUM_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --topk ${TOPK} \
    --eval-batch-size ${EVAL_BATCH} \
    --test-steps ${TEST_STEPS} \
    --seed ${SEED} \
    --max-entries ${MAX_ENTRIES} \
    --compression-target-tokens ${COMP_TOKENS} \
    --output "${RESULT_DIR}/softcom_llama3_3b"

# --- Gemma-3-4B ---
echo "=== QA | SoftCom | Gemma-3-4B ==="
python run_qa_attack.py \
    --data "${DATA_DIR}/squad_qa_filtered_llmlingua1_max900.json" \
    --compressor gemma-3-4b \
    --surrogate-model google/gemma-3-4b-it \
    --num-steps ${NUM_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --topk ${TOPK} \
    --eval-batch-size ${EVAL_BATCH} \
    --test-steps ${TEST_STEPS} \
    --seed ${SEED} \
    --max-entries ${MAX_ENTRIES} \
    --compression-target-tokens ${COMP_TOKENS} \
    --output "${RESULT_DIR}/softcom_gemma3_4b"

echo "=== All QA attacks completed ==="
