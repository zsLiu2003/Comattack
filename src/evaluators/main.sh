# CUDA_VISIBLE_DEVICES=0,1 python /home/lzs/Comattack/src/evaluators/main.py /home/lzs/Comattack/src/data/replaced_confused_recommendation.json
# CUDA_VISIBLE_DEVICES=0,1 python /home/lzs/Comattack/src/evaluators/main.py /home/lzs/Comattack/src/data/new_keywords_decrease_3.json

CUDA_VISIBLE_DEVICES=2,3 python /home/lzs/Comattack/src/evaluators/main_keyword.py /home/lzs/Comattack/src/data/new_keywords_decrease_3.json
# CUDA_VISIBLE_DEVICES=2,3 python /home/lzs/Comattack/src/evaluators/main_keyword.py /home/lzs/Comattack/src/data/replaced_ppl_adjective_increase.json
# CUDA_VISIBLE_DEVICES=2,3 python /home/lzs/Comattack/src/evaluators/main_keyword.py /home/lzs/Comattack/src/data/replaced_ppl_synonym_decrease.json
# CUDA_VISIBLE_DEVICES=2,3 python /home/lzs/Comattack/src/evaluators/main_keyword.py /home/lzs/Comattack/src/data/replaced_ppl_synonym_increase.json  
