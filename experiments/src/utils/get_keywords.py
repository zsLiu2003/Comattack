from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import csv
import os
from tqdm import tqdm
from datasets import load_dataset
from src.data.data_process import CompressionDataset
from src.utils.get_prompt import get_keywords_prompt

import argparse
import torch


def get_parser():

    parser = argparse.ArgumentParser(description="The config of get keywords from demos")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="/opt/model/Qwen3-32B",
        help="name of extraction model",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/lzs/compressionattack/experiments/src/data/data.json",
        help="path of dataset with target"
    )

    parser.add_argument(
        "--device",
        default="cuda: 7" if torch.cuda.is_available() else "cpu",
        help="device map",
    )
    
    # parser.add_argument(
    #     "--prompt_path",
    #     default="/home/lzs/compressionattack/experiments/src/data/get_keywords_prompt.txt",
    #     type=str,
    #     help="path of the prompt"
    # )
    return parser

def get_keywords():

    args = get_parser()
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name,torch_dtype=torch.bfloat16,device_map=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    dataset = CompressionDataset(dataset=dataset)

    prompt = get_keywords_prompt()
    
    for data in tqdm(dataset):

        user_input = ""
        for key, value in data.items():
            if key == "requirements":
                user_input += f"{key}: {value}"
            elif "demo" in key:
                user_input += f",\n{key}: {value}"
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        model_inputs = tokenizer([text], return_tensors='pt').to(model.device)
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768,
        )

        output_ids = output_ids[0][len(model_inputs.input_ids[0]):].tolist()
        try:
    # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens = True)
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True)
    
     