from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import json
from datasets import load_dataset
import argparse

from llmlingua import PromptCompressor
from src.data.data_process import get_common_compression_dataset, get_compression_dataset

# from src.utils.verify_PPL import get_PPL

def get_sequence_length_and_output(tokenizer, text):

    # le_list = []
    # output_list = []
    prompt = ""
    for key, value in text.items():
        if "demo" in key:
            prompt += value
            # le_list.append(tokenizer(text=value,return_tensors="pt")["input_ids"].size(-1))
            # output_list.append(prompt)
    le = tokenizer(text=prompt, return_tensors="pt")["input_ids"].size(-1)
    output = prompt    
    # return [input_ids1.size(-1), input_ids2.size(-1)]
    return le,output

def compress_text(examples, llmlingua_model=None, tokenizer=None,target_token=20):

    le, output = get_sequence_length_and_output(
        tokenizer=tokenizer,
        text=examples,
    )
    compressed_text = llmlingua_model.compress_prompt(
        output,
        instruction="",
        question="",
        target_token = target_token,
    )
    return {
        "compressed_text": compressed_text["compressed_prompt"],
        "original_text": output
    }

def get_compressed_text(model_name=None, dataset=None, device="cpu", target_token=50, output_path=None):
    
    # if "demo_6" in dataset.column_names:
    #     dataset.remove_columns("demo_6")
    compression_model = PromptCompressor(
        model_name=model_name,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    compressed_dataset = dataset.map(compress_text, fn_kwargs={
        "llmlingua_model": compression_model,
        "tokenizer": tokenizer,
        "target_token": target_token,
    },
    )
    # dataset = get_common_compression_dataset(dataset=dataset)
    # compressed_dataset = compressed_dataset["compressed"]
    # for data, compressed_data in zip(dataset, compressed_dataset):
    #     for i, prompt in enumerate(compressed_data):
    #         key = f"demo_{i+1}"
    #         print(len(data[key]), len(prompt["compressed_prompt"]))
    #         data[key] = prompt["compressed_prompt"]
    
    # # if "demo_6" in dataset.column_names:
    # #     dataset.remove_columns("demo_6")
    # print(type(dataset))
    
    """
    for data, compressed_data in zip(dataset, compressed_dataset):
        data["compressed_text"] = compressed_data["compressed_prompt"]
    """

    # if "demo_6" in dataset.column_names:
    #     dataset.remove_columns("demo_6")
    print(compressed_dataset)
    # with open(output_path, 'w', encoding='utf-8') as file:
    #     json.dump(dataset.to_list(), file, indent=4)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(compressed_dataset.to_list(), file, indent=4)
        
    return compressed_dataset