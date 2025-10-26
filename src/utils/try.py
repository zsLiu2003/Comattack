
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

    le_list = []
    output_list = []
    for key, value in text.items():
        if "demo" in key and key != "demo_6":
            print(f"Debugging v: key={key}, type(v)={type(value)}, value={repr(value)}")
            le_list.append(tokenizer(text=value,return_tensors="pt")["input_ids"].size(-1))
            output_list.append(value)
    # return [input_ids1.size(-1), input_ids2.size(-1)]
    return le_list, output_list

def compress_text(examples, llmlingua_model=None, tokenizer=None):

    le, output = get_sequence_length_and_output(
        tokenizer=tokenizer,
        text=examples,
    )
    
    return {
        "compressed" :[llmlingua_model.compress_prompt(
        output[i],
        target_token=le[i]
    ) for i in range(len(le)) ]
    }

def get_compressed_text(model_name=None, dataset=None, device="cpu"):
    
    compression_model = PromptCompressor(
        model_name=model_name,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    compressed_dataset = dataset.map(compress_text, fn_kwargs={
        "llmlingua_model": compression_model,
        "tokenizer": tokenizer,
    },
    )
    dataset = get_common_compression_dataset(dataset=dataset)
    compressed_dataset = compressed_dataset["compressed"]
    for data, compressed_data in zip(dataset, compressed_dataset):
        for i, prompt in enumerate(compressed_data):
            key = f"demo_{i+1}"
            data[key] = prompt["compressed_prompt"]
            
    return dataset


dataset = load_dataset("json", data_files="src/data/data.json", split="train")
dataset = get_common_compression_dataset(dataset=dataset)

get_compressed_text(
    model_name="models/gpt2-dolly",
    dataset=dataset,
    device="cuda:7"
)