compression_model_name="/opt/model/models/gpt2-dolly"
# compression_model_name="/opt/model/models/Llama-2-7b-chat-hf"

CUDA_VISIBLE_DEVICES=4,5 python /home/lzs/Comattack/src/utils/get_ppl.py $compression_model_name