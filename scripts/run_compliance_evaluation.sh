python scripts/run_full_evaluation.py \
  --queries results/systemcheck_sft/adversarial_queries_merged_bert.json \
  --prompts results/phase3.5/filtered_prompts/prompts_original_bert_512.json \
  --compression results/phase3.5/compressed/original_bert_compression.json \
  --output results/systemcheck_sft/behavioral/compliance_Qwen_Qwen3_14B_bert_compression.json \
  --model Qwen/Qwen3-14B \
  --target-url http://localhost:8000/v1 \
  --judge-url http://localhost:8001/v1 \
  --max-workers 32


python scripts/run_full_evaluation.py \
  --queries results/systemcheck_sft/adversarial_queries_merged_gpt2.json \
  --prompts results/phase3.5/filtered_prompts/prompts_original_gpt2_1024.json \
  --compression results/phase3.5/compressed/original_gpt2_compression.json \
  --output results/systemcheck_sft/behavioral/compliance_Qwen_Qwen3_14B_gpt2_compression.json \
  --model Qwen/Qwen3-14B \
  --target-url http://localhost:8000/v1 \
  --judge-url http://localhost:8001/v1 \
  --max-workers 32

python scripts/run_full_evaluation.py \
  --queries results/systemcheck_sft/adversarial_queries_merged_gpt2.json \
  --prompts results/phase3.5/filtered_prompts/prompts_original_gpt2_1024.json \
  --allow-empty-prompt \
  --output results/systemcheck_sft/behavioral/compliance_Qwen_Qwen3_14B_gpt2_no_prompt.json \
  --model Qwen/Qwen3-14B \
  --target-url http://localhost:8000/v1 \
  --judge-url http://localhost:8001/v1 \
  --max-workers 32

------------------------------------------------------------------------------------------------
# run no system prompt + full system prompt with shortened prompts for phase3.5
python scripts/run_full_evaluation.py \
  --queries results/phase3.5/adversarial_queries_filtered_gpt2.json \
  --prompts results/phase3.5/filtered_prompts/prompts_short_gpt2_1024.json \
  --output results/systemcheck_sft/behavioral/compliance_meta-llama_Llama-3.1-8B-Instruct_shortened_gpt2_compression_filtered_queries.json \
  --model Qwen/Qwen3-14B \
  --allow-empty-prompt \
  --target-url http://localhost:8000/v1 \
  --judge-url http://localhost:8001/v1 \
  --max-workers 32

# run LLMLingua1 compression with shortened prompts for phase3.5
python scripts/run_full_evaluation.py \
  --queries results/phase3.5/adversarial_queries_filtered_gpt2.json \
  --prompts results/phase3.5/filtered_prompts/prompts_short_gpt2_1024.json \
  --compression results/phase3.5/compressed/shortened_gpt2_compression.json \
  --output results/systemcheck_sft/behavioral/compliance_meta-llama_Llama-3.1-8B-Instruct_shortened_gpt2_compression_filtered_queries.json \
  --model Qwen/Qwen3-14B \
  --target-url http://localhost:8000/v1 \
  --judge-url http://localhost:8001/v1 \
  --max-workers 32

# run LLMLingua2 compression with shortened prompts for phase3.5
python scripts/run_full_evaluation.py \
  --queries results/phase3.5/adversarial_queries_filtered_bert.json \
  --prompts results/phase3.5/filtered_prompts/prompts_short_bert_512.json \
  --compression results/phase3.5/compressed/shortened_bert_compression.json \
  --output results/systemcheck_sft/behavioral/compliance_meta-llama_Llama-3.1-8B-Instruct_shortened_bert_compression_filtered_queries.json \
  --model Qwen/Qwen3-14B \
  --target-url http://localhost:8000/v1 \
  --judge-url http://localhost:8001/v1 \
  --max-workers 32