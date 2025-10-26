from src.evaluators.product_recommendation_test import Product_recommendation
from datasets import load_dataset   
import json
import torch
from tqdm import tqdm
import sys
from src.evaluators.product_recommendation_test import Product_recommendation

def main(dataset_path):
    
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    model_name = "models/gpt2-dolly"
    phrase_model_name = "models/Qwen3-32B"
    
    flag = None
    if "increase" in dataset_path:
        flag = True
    elif "decrease" in dataset_path:
        flag = False
    if "keyword" in dataset_path:
        flag = "keyword"
    
    product_recommendation = Product_recommendation(
        compression_model_name=model_name,
        dataset_path=dataset_path,
        question_dataset_path=dataset_path,
        device="cuda:0"
    )
    
    if flag == "keyword":
        product_recommendation.keywords_test(
            dataset=dataset,
            model_name=model_name,
            phrase_model_name=phrase_model_name,
            flag=flag,
        )
    else:
        product_recommendation.token_level_test(
            dataset=dataset,
            model_name=model_name,
            phrase_model_name=phrase_model_name,
            flag=flag
        )

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python main_keyword.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    main(dataset_path)

