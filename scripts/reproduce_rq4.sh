set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    EXTRA_ARGS+=" $1"
    shift
done

mkdir -p results/rq4

# =====================================================================
# Case Study 1: VSCode Cline -- Guardrail Corruption
# =====================================================================
echo "===== Case Study 1: VSCode Cline ====="
python case_studies/case_study_cline.py \
    --output results/rq4/case_study_1_cline/ \
    $EXTRA_ARGS

echo ""

# =====================================================================
# Case Study 2: LangChain + Ollama -- Tool Selection Manipulation
# =====================================================================
echo "===== Case Study 2: LangChain + Ollama ====="
python case_studies/case_study_langchain_ollama.py \
    --output results/rq4/case_study_2_langchain/ \
    $EXTRA_ARGS

echo ""
echo "===== RQ4 reproduction complete ====="
echo "Results saved under results/rq4/"
echo ""
echo "To view results:"
echo "  cat results/rq4/case_study_1_cline/case_study_1_cline_results.json | python -m json.tool | head -50"
echo "  cat results/rq4/case_study_2_langchain/case_study_2_langchain_results.json | python -m json.tool | head -50"
