from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import csv
import json
from tqdm import tqdm
from datasets import load_dataset
import argparse
from src.data.data_process import CompressionCommonDataset
from src.utils.get_compressed_text import get_compressed_text
import deepspeed
from src.utils.verify_PPL import get_PPL
from src.utils.get_best_output import get_best_output

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compression_model",
        type=str,
        default="/opt/lzs/models/gpt2-dolly",
        help="path of model to calculte the PPL of every token"
    )

    parser.add_argument(
        "--large_model",
        type=str,
        default="/opt/model/models/Llama-2-7b-chat-hf",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/lzs/compressionattack/experiments/src/data/data.json",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="the number of selected tokens with a high PPL"
    )

    return parser

# def pure_dataset(examples):
    
#     for key,value in examples.items():
        
def replace_demos(example, idx, compressed_dataset, demo_keys):
    
    compressed_prompt = compressed_dataset[idx]
    for key in demo_keys:
        example[key] = compressed_prompt[key]
    
    return example
       

def get_ppl():

    parser = get_parser()
    args = parser.parse_args()
    
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    dataset = CompressionCommonDataset(dataset=dataset)
    compressed_dataset = get_compressed_text(
        model_name=args.compression_model,
        dataset=dataset,
        device="cuda:0",
    )
    demo_keys = [key for key in dataset.keys() if "demo" in key]
    dataset.map(
        replace_demos,
        with_indics=True,
        batched=False,
        fn_kwargs={
            "compressed_dataset": compressed_dataset,
            "demo_keys": demo_keys, 
        }
    )
    output_data_path = "/home/lzs/compressionattack/experiments/src/data/data_with_compressed.json"
    with open(output_data_path, "w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=4)
    
    dataset = get_best_output(
        other_dataset=dataset,
        data_with_target_path="/home/lzs/compressionattack/experiments/src/data/data_with_compressed_target.json"
        )
    
    
            

    remain_columns = [key for key in dataset.keys() if "demo" in key]
    remain_dataset = dataset.select_columns(remain_columns)

    for i in tqdm(range(len(dataset)), desc="Get the PPL of every token: "):
        data = remain_dataset[i]
        compressed_data=compressed_data[i]
        assert len(data) == len(compressed_data), "the length of original data is not equal to the compressed data"
        
        for original_demo, compressed_output in zip(data, compressed_data):
            
    
    