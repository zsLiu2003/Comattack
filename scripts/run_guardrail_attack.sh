#!/bin/bash
# COMA – System Prompt Corruption (Task 3) attack launcher
#
# Usage:
#   bash scripts/run_guardrail_attack.sh

set -euo pipefail

DATA="data/guardrail_dataset.json"
COMPRESSOR="llmlingua1"
SURROGATE="NousResearch/Llama-2-7b-hf"
OUTPUT="results/guardrail_${COMPRESSOR}/"
NUM_STEPS=200
BATCH_SIZE=256
TOPK=64

python run_guardrail_attack.py \
    --data "$DATA" \
    --compressor "$COMPRESSOR" \
    --surrogate-model "$SURROGATE" \
    --num-steps $NUM_STEPS \
    --batch-size $BATCH_SIZE \
    --topk $TOPK \
    --output "$OUTPUT"
