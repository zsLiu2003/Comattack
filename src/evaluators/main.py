from src.evaluators.product_recommendation_test import Product_recommendation
from datasets import load_dataset   
import json
import torch
from tqdm import tqdm

def main():
    compression_model_name = "/opt/model/models/gpt2-dolly"
    dataset_path_list = [
        "/home/lzs/Comattack/src/data/replaced_ppl_adjective_increase.json",
        "/home/lzs/Comattack/src/data/replaced_ppl_adjective_decrease.json",
        "/home/lzs/Comattack/src/data/replaced_ppl_noun_increase.json",
        "/home/lzs/Comattack/src/data/replaced_ppl_noun_decrease.json",
    ]
    question_dataset_path = "/home/lzs/Comattack/src/data/data.json"
    
    for dataset_path in tqdm(dataset_path_list):
        print(f"-----------------------Processing dataset: {dataset_path}------------------------")
        test = Product_recommendation(
            compression_model_name=compression_model_name,
            dataset_path=dataset_path,
            question_dataset_path=question_dataset_path,
            device="cuda:0"
        )
        test.demo_level_test()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()