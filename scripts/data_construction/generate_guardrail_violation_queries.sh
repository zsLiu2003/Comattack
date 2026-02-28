# the command is to generate adversarial queries for guardrails (leaked system prompts)
python scripts/generate_adversarial_queries_v2.py \
    --input_path "data/extracted_guardrails_reduced_filtered.json" \
    --output_path "data/guardrail_violation_queries_for_leaked_system_prompt.json" \
    --model "Qwen/Qwen3-32B" \
    --target-url "http://localhost:8001/v1" \
    --system-prompt-file "data/adversarial_query_generation_system_prompt.txt" \
    --max-workers 128


# the command is to generate adversarial queries for guardrails (hand_written)
python scripts/generate_adversarial_queries_v2.py \
    --input_path "data/handwritten_system_prompt_guardrails.json" \
    --output_path "data/guardrail_violation_queries_for_handwritten_system_prompt.json" \
    --model "Qwen/Qwen3-32B" \
    --target-url "http://localhost:8001/v1" \
    --system-prompt-file "data/adversarial_query_generation_for_handwritten_system_prompt.txt" \
    --max-workers 32