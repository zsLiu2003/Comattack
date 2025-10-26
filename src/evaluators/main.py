from src.evaluators.product_recommendation_test import Product_recommendation
from datasets import load_dataset   
import json
import torch
from tqdm import tqdm
import sys

def main(dataset_path):
    compression_model_name = "models/gpt2-dolly"
    
    question_dataset_path = "src/data/data.json"
    
    # for dataset_path in tqdm(dataset_path_list):
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
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "src/data/increase_demo.json"
    main(dataset_path=dataset_path)

