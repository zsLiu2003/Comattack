from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import json
from datasets import load_dataset
import argparse

from llmlingua import PromptCompressor
# from src.utils.verify_PPL import get_PPL

def get_sequence_length_and_output(tokenizer, text):

    le_list = []
    output_list = []
    for k ,v in text:
        if "demo" in k:
            le_list.append(tokenizer(v, return_tensors='pt')["input_ids"].size(-1))
            output_list.append(v)
    # return [input_ids1.size(-1), input_ids2.size(-1)]
    return le_list, output_list

def compress_text(examples, llmlingua_model=None, tokenizer=None):

    le, output = get_sequence_length_and_output(
        tokenizer=tokenizer,
        text=examples,
    )
    
    return [llmlingua_model.compress_prompt(
        output[i],
        target_token=le[i]
    ) for i in range(len(le)) ]

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
    
    return compressed_dataset