compression_model_name="models/gpt2-dolly"
# compression_model_name="models/Llama-2-7b-chat-hf"

CUDA_VISIBLE_DEVICES=4,5 python ./src/utils/get_ppl.py $compression_model_name