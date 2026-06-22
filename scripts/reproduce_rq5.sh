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

echo "===== RQ5: Defense Evaluation ====="
echo "NOTE: This requires attack results from RQ1."
echo "      Run scripts/reproduce_rq1.sh first if needed."
echo ""

mkdir -p results/rq5

# =====================================================================
# Defense 1: Perplexity-based Detection
# =====================================================================
echo "--- Defense 1: Perplexity-based Detection ---"

for TASK_CFG in \
    "qa:results/rq1/qa/extractive_llmlingua1/qa_extractive_results.jsonl" \
    "pref:results/rq1/pref/extractive_llmlingua1/pref_extractive_results.jsonl" \
    "spc:results/rq1/spc/extractive_llmlingua1/guardrail_extractive_results.jsonl"; do

    IFS=: read -r TASK RESULTS_FILE <<< "$TASK_CFG"

    if [ -f "$RESULTS_FILE" ] || $DRY_RUN; then
        run_cmd python -c "
from comattack.defense.ppl_detector import PPLDetector
import json

detector = PPLDetector()

results = []
with open('${RESULTS_FILE}') as f:
    for line in f:
        entry = json.loads(line.strip())
        attacked = entry.get('attacked_context') or entry.get('attacked_prompt', '')
        original = entry.get('context') or entry.get('system_prompt', '')
        if attacked and original:
            score = detector.detect(original, attacked)
            results.append(score)

import os, json as j
os.makedirs('results/rq5/ppl_detection/${TASK}', exist_ok=True)
with open('results/rq5/ppl_detection/${TASK}/scores.json', 'w') as f:
    j.dump({'task': '${TASK}', 'n': len(results), 'scores': results}, f, indent=2)
print(f'  PPL detection on ${TASK}: {len(results)} entries scored')
"
    else
        echo "  [SKIP] ${TASK} -- results file not found: ${RESULTS_FILE}"
    fi
done

# =====================================================================
# Defense 2: BLEU-based Detection
# =====================================================================
echo ""
echo "--- Defense 2: BLEU-based Detection ---"

for TASK_CFG in \
    "qa:results/rq1/qa/extractive_llmlingua1/qa_extractive_results.jsonl" \
    "pref:results/rq1/pref/extractive_llmlingua1/pref_extractive_results.jsonl" \
    "spc:results/rq1/spc/extractive_llmlingua1/guardrail_extractive_results.jsonl"; do

    IFS=: read -r TASK RESULTS_FILE <<< "$TASK_CFG"

    if [ -f "$RESULTS_FILE" ] || $DRY_RUN; then
        run_cmd python -c "
from comattack.defense.bleu_detector import BLEUDetector
import json

detector = BLEUDetector()

results = []
with open('${RESULTS_FILE}') as f:
    for line in f:
        entry = json.loads(line.strip())
        attacked = entry.get('attacked_context') or entry.get('attacked_prompt', '')
        original = entry.get('context') or entry.get('system_prompt', '')
        if attacked and original:
            score = detector.detect(original, attacked)
            results.append(score)

import os, json as j
os.makedirs('results/rq5/bleu_detection/${TASK}', exist_ok=True)
with open('results/rq5/bleu_detection/${TASK}/scores.json', 'w') as f:
    j.dump({'task': '${TASK}', 'n': len(results), 'scores': results}, f, indent=2)
print(f'  BLEU detection on ${TASK}: {len(results)} entries scored')
"
    else
        echo "  [SKIP] ${TASK} -- results file not found: ${RESULTS_FILE}"
    fi
done

# =====================================================================
# Defense 3: LLM-based Detection
# =====================================================================
echo ""
echo "--- Defense 3: LLM-based Detection ---"

for TASK_CFG in \
    "qa:results/rq1/qa/extractive_llmlingua1/qa_extractive_results.jsonl" \
    "pref:results/rq1/pref/extractive_llmlingua1/pref_extractive_results.jsonl" \
    "spc:results/rq1/spc/extractive_llmlingua1/guardrail_extractive_results.jsonl"; do

    IFS=: read -r TASK RESULTS_FILE <<< "$TASK_CFG"

    if [ -f "$RESULTS_FILE" ] || $DRY_RUN; then
        run_cmd python -c "
from comattack.defense.llm_detector import LLMDetector
import json

detector = LLMDetector()

results = []
with open('${RESULTS_FILE}') as f:
    for line in f:
        entry = json.loads(line.strip())
        attacked = entry.get('attacked_context') or entry.get('attacked_prompt', '')
        original = entry.get('context') or entry.get('system_prompt', '')
        if attacked and original:
            score = detector.detect(original, attacked)
            results.append(score)

import os, json as j
os.makedirs('results/rq5/llm_detection/${TASK}', exist_ok=True)
with open('results/rq5/llm_detection/${TASK}/scores.json', 'w') as f:
    j.dump({'task': '${TASK}', 'n': len(results), 'scores': results}, f, indent=2)
print(f'  LLM detection on ${TASK}: {len(results)} entries scored')
"
    else
        echo "  [SKIP] ${TASK} -- results file not found: ${RESULTS_FILE}"
    fi
done

echo ""
echo "===== RQ5 reproduction complete ====="
echo "Results saved under results/rq5/"
