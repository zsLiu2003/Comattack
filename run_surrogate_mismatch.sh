set -euo pipefail
cd "$(dirname "$0")"

# ── defaults ─────────────────────────────────────────────────────────
MAX_PARALLEL=1
DRY_RUN=false
RESUME=true
GPUS=""

OUTPUT="results/rq3_mismatch"
LOG_DIR="${OUTPUT}/logs"
DATA_PREF="data/pref_manipulation_filtered.json"
DATA_QA="data/squad_qa_filtered.json"
DATA_SPC="data/guardrail_dataset.json"
BACKEND="meta-llama/Llama-3.1-8B-Instruct"
STEPS=500
SEED=42

TASKS=(prom deg qa spc)

# ── argument parsing ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)     DRY_RUN=true; shift ;;
        --no-resume)   RESUME=false; shift ;;
        --parallel)    MAX_PARALLEL="$2"; shift 2 ;;
        --gpus)        GPUS="$2"; shift 2 ;;
        --output)      OUTPUT="$2"; LOG_DIR="${OUTPUT}/logs"; shift 2 ;;
        --steps)       STEPS="$2"; shift 2 ;;
        --seed)        SEED="$2"; shift 2 ;;
        --backend)     BACKEND="$2"; shift 2 ;;
        --data-pref)   DATA_PREF="$2"; shift 2 ;;
        --data-qa)     DATA_QA="$2"; shift 2 ;;
        --data-spc)    DATA_SPC="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# split GPU list into array
IFS=',' read -ra GPU_LIST <<< "${GPUS}"
N_GPUS=${#GPU_LIST[@]}

mkdir -p "${LOG_DIR}"

# ── bookkeeping ──────────────────────────────────────────────────────
TOTAL=0
SKIPPED=0
LAUNCHED=0
FAILED=0
PIDS=()
JOB_LABELS=()

safe_name() { echo "$1" | tr '/' '_'; }

is_done() {
    local tgt="$1" surr="$2" task="$3"
    local sj="${OUTPUT}/$(safe_name "$tgt")/$(safe_name "$surr")/${task}/summary.json"
    [[ -f "$sj" ]]
}

# ── job launcher ─────────────────────────────────────────────────────
wait_for_slot() {
    while (( ${#PIDS[@]} >= MAX_PARALLEL )); do
        local new_pids=()
        local new_labels=()
        for i in "${!PIDS[@]}"; do
            if kill -0 "${PIDS[$i]}" 2>/dev/null; then
                new_pids+=("${PIDS[$i]}")
                new_labels+=("${JOB_LABELS[$i]}")
            else
                wait "${PIDS[$i]}" 2>/dev/null || FAILED=$((FAILED + 1))
            fi
        done
        PIDS=("${new_pids[@]+"${new_pids[@]}"}")
        JOB_LABELS=("${new_labels[@]+"${new_labels[@]}"}")
        if (( ${#PIDS[@]} >= MAX_PARALLEL )); then
            sleep 5
        fi
    done
}

launch() {
    local tgt="$1" surr="$2" task="$3"
    TOTAL=$((TOTAL + 1))
    local label="${tgt} | ${surr} | ${task}"

    if $RESUME && is_done "$tgt" "$surr" "$task"; then
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    local cmd="python run_rq3_surrogate_mismatch.py"
    cmd+=" --target-compressor ${tgt}"
    cmd+=" --surrogate-model ${surr}"
    cmd+=" --task ${task}"
    cmd+=" --data-pref ${DATA_PREF}"
    cmd+=" --data-qa ${DATA_QA}"
    cmd+=" --data-spc ${DATA_SPC}"
    cmd+=" --backend-llm ${BACKEND}"
    cmd+=" --num-steps ${STEPS}"
    cmd+=" --seed ${SEED}"
    cmd+=" --output ${OUTPUT}"

    if $DRY_RUN; then
        echo "$cmd"
        return
    fi

    local log_file="${LOG_DIR}/$(safe_name "$tgt")__$(safe_name "$surr")__${task}.log"

    # assign GPU round-robin when GPUs are specified
    local gpu_env=""
    if (( N_GPUS > 0 )); then
        local gpu_idx=$(( LAUNCHED % N_GPUS ))
        gpu_env="CUDA_VISIBLE_DEVICES=${GPU_LIST[$gpu_idx]}"
    fi

    wait_for_slot

    echo "[$(date +%H:%M:%S)] launch ${label}"
    if [[ -n "$gpu_env" ]]; then
        env $gpu_env bash -c "$cmd" > "$log_file" 2>&1 &
    else
        bash -c "$cmd" > "$log_file" 2>&1 &
    fi
    PIDS+=($!)
    JOB_LABELS+=("$label")
    LAUNCHED=$((LAUNCHED + 1))
}

# ── build grid ───────────────────────────────────────────────────────
echo "=== RQ3 Surrogate Mismatch Grid ==="
echo "  parallel=${MAX_PARALLEL}  gpus=${GPUS:-auto}  resume=${RESUME}"
echo ""

echo "--- Extractive PPL-based (SC, LLMLingua1) ---"
for tgt in selective_context llmlingua1; do
    for surr in gpt2 NousResearch/Llama-2-7b-hf microsoft/phi-2; do
        for task in "${TASKS[@]}"; do launch "$tgt" "$surr" "$task"; done
    done
done

echo "--- Extractive Classifier (LLMLingua2) ---"
for surr in bert-base-uncased microsoft/deberta-v3-large xlm-roberta-large; do
    for task in "${TASKS[@]}"; do launch llmlingua2 "$surr" "$task"; done
done

echo "--- Abstractive (cross-surrogate) ---"
for tgt in llama-3.2-3b qwen3-4b gemma-3-4b; do
    for surr in llama-3.2-3b qwen3-4b gemma-3-4b; do
        for task in "${TASKS[@]}"; do launch "$tgt" "$surr" "$task"; done
    done
done

# ── wait for stragglers ──────────────────────────────────────────────
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" 2>/dev/null || FAILED=$((FAILED + 1))
done

# ── aggregate ────────────────────────────────────────────────────────
if ! $DRY_RUN; then
    echo ""
    echo "--- Aggregating results into Table 5 ---"
    python run_rq3_surrogate_mismatch.py --aggregate --output "${OUTPUT}"
fi

# ── summary ──────────────────────────────────────────────────────────
echo ""
echo "=== Summary ==="
echo "  total configs : ${TOTAL}"
echo "  skipped (done): ${SKIPPED}"
echo "  launched      : ${LAUNCHED}"
echo "  failed        : ${FAILED}"
echo "  logs          : ${LOG_DIR}/"
echo "Done."
