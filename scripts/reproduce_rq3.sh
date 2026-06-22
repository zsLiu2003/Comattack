set -euo pipefail
cd "$(dirname "$0")/.."

PART="all"
PARALLEL=1
DRY_RUN=false
GPUS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --part) PART="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# =====================================================================
# Part A: Surrogate Mismatch Grid
# =====================================================================
if [[ "$PART" == "all" || "$PART" == "a" ]]; then
    echo "===== RQ3 Part A: Surrogate Mismatch Grid ====="

    MISMATCH_ARGS=(
        --output results/rq3_mismatch
        --data-pref data/ats/pref_manipulation_filtered_llmlingua1_max900.json
        --data-qa data/qa/squad_qa_filtered_llmlingua1_max900.json
        --data-spc data/system_prompt/leaked_system_prompt_guardrails.json
    )

    if [[ -n "$GPUS" ]]; then
        MISMATCH_ARGS+=(--gpus "$GPUS")
    fi

    if $DRY_RUN; then
        bash run_rq3_surrogate_mismatch.sh \
            --parallel "$PARALLEL" \
            --dry-run \
            "${MISMATCH_ARGS[@]}"
    else
        bash run_rq3_surrogate_mismatch.sh \
            --parallel "$PARALLEL" \
            "${MISMATCH_ARGS[@]}"
    fi

    echo ""
    echo "Surrogate mismatch results: results/rq3_mismatch/"
fi

# =====================================================================
# Part B: Critical Token Retention
# =====================================================================
if [[ "$PART" == "all" || "$PART" == "b" ]]; then
    echo ""
    echo "===== RQ3 Part B: Critical Token Retention ====="
    echo "NOTE: This requires attack results from RQ1 to exist."
    echo "      Run scripts/reproduce_rq1.sh first if needed."

    run_cmd() {
        if $DRY_RUN; then echo "$@"; else echo ">>> $@"; "$@"; fi
    }

    # Run retention analysis for each available (compressor, task) pair
    for COMP in llmlingua1 llmlingua2; do
        for TASK_CFG in \
            "qa:results/rq1/qa/extractive_${COMP}/qa_extractive_results.jsonl" \
            "prom:results/rq1/pref/extractive_${COMP}/pref_extractive_results.jsonl" \
            "spc:results/rq1/spc/extractive_${COMP}/guardrail_extractive_results.jsonl"; do

            IFS=: read -r TASK RESULTS_FILE <<< "$TASK_CFG"

            if [ -f "$RESULTS_FILE" ] || $DRY_RUN; then
                run_cmd python run_token_retention.py \
                    --attack-results "$RESULTS_FILE" \
                    --compressor "$COMP" \
                    --task "$TASK" \
                    --compression-rate 0.6 \
                    --output results/rq3_retention/
            else
                echo "  [SKIP] ${COMP}/${TASK} -- results file not found: ${RESULTS_FILE}"
                echo "         Run scripts/reproduce_rq1.sh first."
            fi
        done
    done

    # Aggregate retention results
    if ! $DRY_RUN; then
        echo ""
        echo "--- Aggregating token retention results ---"
        python run_token_retention.py --aggregate --output results/rq3_retention/
    fi

    echo ""
    echo "Token retention results: results/rq3_retention/"
fi

echo ""
echo "===== RQ3 reproduction complete ====="
