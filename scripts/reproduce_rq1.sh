set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=false
TASK_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --task) TASK_FILTER="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

run_cmd() {
    if $DRY_RUN; then
        echo "$@"
    else
        echo ">>> $@"
        "$@"
    fi
}

# ── Shared parameters ───────────────────────────────────────────────
NUM_STEPS=500
BATCH_SIZE=256
TOPK=64
EVAL_BATCH=128
TEST_STEPS=20
SEED=42
COMP_TOKENS=200

# =====================================================================
# Task 1: Agent Tool Selection (ATS) -- Preference Manipulation
# =====================================================================
if [[ -z "$TASK_FILTER" || "$TASK_FILTER" == "ats" || "$TASK_FILTER" == "pref" ]]; then
    echo ""
    echo "===== RQ1: Agent Tool Selection (ATS) ====="

    # Extractive-based
    for comp_cfg in \
        "llmlingua1:data/ats/pref_manipulation_filtered_llmlingua1_max900.json:NousResearch/Llama-2-7b-hf" \
        "llmlingua2:data/ats/pref_manipulation_filtered_llmlingua2_max460.json:microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"; do
        IFS=: read -r COMP DATA SURR <<< "$comp_cfg"
        run_cmd python run_pref_attack.py \
            --data "$DATA" --compressor "$COMP" --surrogate-model "$SURR" \
            --num-steps $NUM_STEPS --batch-size $BATCH_SIZE --topk $TOPK \
            --eval-batch-size $EVAL_BATCH --test-steps $TEST_STEPS --seed $SEED \
            --output "results/rq1/pref/extractive_${COMP}"
    done

    # Summarize-based
    for comp_cfg in \
        "qwen3-4b:Qwen/Qwen3-4B" \
        "llama-3.2-3b:meta-llama/Llama-3.2-3B-Instruct" \
        "gemma-3-4b:google/gemma-3-4b-it"; do
        IFS=: read -r COMP SURR <<< "$comp_cfg"
        run_cmd python run_pref_attack.py \
            --data data/ats/pref_manipulation_filtered_llmlingua1_max900.json \
            --compressor "$COMP" --surrogate-model "$SURR" \
            --num-steps $NUM_STEPS --batch-size $BATCH_SIZE --topk $TOPK \
            --eval-batch-size $EVAL_BATCH --test-steps $TEST_STEPS --seed $SEED \
            --compression-target-tokens $COMP_TOKENS \
            --output "results/rq1/pref/summarize_based_${COMP}"
    done
fi

# =====================================================================
# Task 2: Question Answering (QA)
# =====================================================================
if [[ -z "$TASK_FILTER" || "$TASK_FILTER" == "qa" ]]; then
    echo ""
    echo "===== RQ1: Question Answering (QA) ====="

    # Extractive-based
    for comp_cfg in \
        "llmlingua1:data/qa/squad_qa_filtered_llmlingua1_max900.json:NousResearch/Llama-2-7b-hf" \
        "llmlingua2:data/qa/squad_qa_filtered_llmlingua2_max460.json:microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"; do
        IFS=: read -r COMP DATA SURR <<< "$comp_cfg"
        run_cmd python run_qa_attack.py \
            --data "$DATA" --compressor "$COMP" --surrogate-model "$SURR" \
            --num-steps $NUM_STEPS --batch-size $BATCH_SIZE --topk $TOPK \
            --eval-batch-size $EVAL_BATCH --test-steps $TEST_STEPS --seed $SEED \
            --output "results/rq1/qa/extractive_${COMP}"
    done

    # Summarize-based
    for comp_cfg in \
        "qwen3-4b:Qwen/Qwen3-4B" \
        "llama-3.2-3b:meta-llama/Llama-3.2-3B-Instruct" \
        "gemma-3-4b:google/gemma-3-4b-it"; do
        IFS=: read -r COMP SURR <<< "$comp_cfg"
        run_cmd python run_qa_attack.py \
            --data data/qa/squad_qa_filtered_llmlingua1_max900.json \
            --compressor "$COMP" --surrogate-model "$SURR" \
            --num-steps $NUM_STEPS --batch-size $BATCH_SIZE --topk $TOPK \
            --eval-batch-size $EVAL_BATCH --test-steps $TEST_STEPS --seed $SEED \
            --compression-target-tokens $COMP_TOKENS \
            --output "results/rq1/qa/summarize_based_${COMP}"
    done
fi

# =====================================================================
# Task 3: System Prompt Corruption (SPC)
# =====================================================================
if [[ -z "$TASK_FILTER" || "$TASK_FILTER" == "spc" ]]; then
    echo ""
    echo "===== RQ1: System Prompt Corruption (SPC) ====="

    # Extractive-based
    for comp_cfg in \
        "llmlingua1:data/system_prompt/guardrail_violation_queries_for_leaked_system_prompt_filtered_llmlingua1_v1.json:NousResearch/Llama-2-7b-hf" \
        "llmlingua2:data/system_prompt/guardrail_violation_queries_for_leaked_system_prompt_filtered_llmlingua2_v1.json:microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"; do
        IFS=: read -r COMP DATA SURR <<< "$comp_cfg"
        run_cmd python run_guardrail_attack.py \
            --data "$DATA" --compressor "$COMP" --surrogate-model "$SURR" \
            --num-steps $NUM_STEPS --batch-size $BATCH_SIZE --topk $TOPK \
            --eval-batch-size $EVAL_BATCH --test-steps $TEST_STEPS --seed $SEED \
            --output "results/rq1/spc/extractive_${COMP}"
    done

    # Summarize-based
    for comp_cfg in \
        "qwen3-4b:Qwen/Qwen3-4B" \
        "llama-3.2-3b:meta-llama/Llama-3.2-3B-Instruct" \
        "gemma-3-4b:google/gemma-3-4b-it"; do
        IFS=: read -r COMP SURR <<< "$comp_cfg"
        run_cmd python run_guardrail_attack.py \
            --data data/system_prompt/leaked_system_prompt_guardrails.json \
            --compressor "$COMP" --surrogate-model "$SURR" \
            --num-steps $NUM_STEPS --batch-size $BATCH_SIZE --topk $TOPK \
            --eval-batch-size $EVAL_BATCH --test-steps $TEST_STEPS --seed $SEED \
            --compression-target-tokens $COMP_TOKENS \
            --output "results/rq1/spc/summarize_based_${COMP}"
    done
fi

echo ""
echo "===== RQ1 reproduction complete ====="
echo "Results saved under results/rq1/"
