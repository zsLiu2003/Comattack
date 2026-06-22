set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

run_cmd() {
    if $DRY_RUN; then echo "$@"; else echo ">>> $@"; "$@"; fi
}

# ── Parameters ───────────────────────────────────────────────────────
NUM_STEPS=500
BATCH_SIZE=256
TOPK=64
EVAL_BATCH=128
TEST_STEPS=20
SEED=42

# =====================================================================
# RQ2a: Compression Budget Sweep
# =====================================================================
echo "===== RQ2a: Compression Budget Sweep ====="

# Vary compression rate for LLMLingua-1 on QA task
for RATE in 0.2 0.4 0.6 0.8; do
    echo "--- Budget: rate=${RATE} ---"
    run_cmd python run_qa_attack.py \
        --data data/qa/squad_qa_filtered_llmlingua1_max900.json \
        --compressor llmlingua1 \
        --surrogate-model NousResearch/Llama-2-7b-hf \
        --num-steps $NUM_STEPS --batch-size $BATCH_SIZE --topk $TOPK \
        --eval-batch-size $EVAL_BATCH --test-steps $TEST_STEPS --seed $SEED \
        --output "results/rq2/budget_sweep/qa_llmlingua1_rate${RATE}"
done

# Vary compression rate for LLMLingua-2 on ATS task
for RATE in 0.2 0.4 0.6 0.8; do
    echo "--- Budget: rate=${RATE} ---"
    run_cmd python run_pref_attack.py \
        --data data/ats/pref_manipulation_filtered_llmlingua2_max460.json \
        --compressor llmlingua2 \
        --surrogate-model microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank \
        --num-steps $NUM_STEPS --batch-size $BATCH_SIZE --topk $TOPK \
        --eval-batch-size $EVAL_BATCH --test-steps $TEST_STEPS --seed $SEED \
        --output "results/rq2/budget_sweep/pref_llmlingua2_rate${RATE}"
done

# =====================================================================
# RQ2b: Backend LLM Sweep
# =====================================================================
echo ""
echo "===== RQ2b: Backend LLM Sweep ====="

# Test with different backend LLMs (requires vLLM or API access)
for BACKEND in \
    "meta-llama/Llama-3.1-8B-Instruct" \
    "Qwen/Qwen3-8B" \
    "microsoft/phi-4"; do

    SAFE_NAME=$(echo "$BACKEND" | tr '/' '_')
    echo "--- Backend: ${BACKEND} ---"

    # QA task with LLMLingua-1
    run_cmd python run_qa_attack.py \
        --data data/qa/squad_qa_filtered_llmlingua1_max900.json \
        --compressor llmlingua1 \
        --surrogate-model NousResearch/Llama-2-7b-hf \
        --num-steps $NUM_STEPS --batch-size $BATCH_SIZE --topk $TOPK \
        --eval-batch-size $EVAL_BATCH --test-steps $TEST_STEPS --seed $SEED \
        --output "results/rq2/backend_sweep/qa_llmlingua1_${SAFE_NAME}"
done

echo ""
echo "===== RQ2 reproduction complete ====="
echo "Results saved under results/rq2/"
