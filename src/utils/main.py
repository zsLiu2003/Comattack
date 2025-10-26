from src.utils.get_ppl import get_ppl
from datasets import load_dataset
import json
import argparse

def get_argparse():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Qwen3-32B",
        help="Path/name of the Large Model",
    )

    parser.add_argument(
        "--compression_model_path",
        type=str,
        default="models/gpt2-dolly",
        help="Path/name of the compression model"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="src/data/data.json",
    )

    return parser.parse_args()

def main():

    args = get_argparse()
    
    dataset = load_dataset("json",data_files=args.dataset_path, split="train")

    get_ppl(
        model_path=args.model_path,
        compression_model_path=args.compression_model_path,
        dataset=dataset,
        top_k=20,
        output_path="src/data",
        target_token=20,
    )

if __name__ == "__main__":

    main()