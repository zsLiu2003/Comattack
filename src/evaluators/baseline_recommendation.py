import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmlingua import PromptCompressor
from datasets import load_dataset
from src.data.data_process import get_QA_dataset, get_common_compression_dataset
from src.attacks.baseline_prompt import context_ignore_prompt,direct_injectiion_prompt,escape_characters_prompt
import random


def clean(dataset_path):

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    return dataset

def naive_attack(dataset_path):
    """"""
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = get_common_compression_dataset(dataset=dataset)
    
    dataset_list = []
    for data_entry in dataset:
        data_dict = data_entry.copy()
        for key, value in data_entry.items():
            if "demo" in key:
                context_chars = list(value)
                indices = random.sample(range(len(context_chars)), min(10, len(context_chars)))

                for idx in indices:
                    old_char = context_chars[idx]
                    if old_char.isalpha():
                        context_chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz'.replace(old_char.lower(), ''))
                    else:
                        context_chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')

                attacked_context = ''.join(context_chars)
                new_key = f"replaced_{key}"
                data_dict[new_key] = attacked_context
        
        dataset_list.append(data_dict)
        
    return dataset_list

def direct_injection(dataset_path: str):
    """"""
    prompt = direct_injectiion_prompt()
    
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = get_common_compression_dataset(dataset=dataset)

    dataset_list = []
    for data_entry in dataset:
        data_dict = data_entry.copy()
        for key, value in data_entry.items():
            if "demo" in key:
                attacked_context = value + prompt
                new_key = f"replaced_{key}"
                data_dict[new_key] = attacked_context
        
        dataset_list.append(data_dict)
    
    return dataset_list
        

def escape_characters(dataset_path: str):
    """"""

    prompt = escape_characters_prompt()
    
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = get_common_compression_dataset(dataset=dataset)

    dataset_list = []
    for data_entry in dataset:
        data_dict = data_entry.copy()
        for key, value in data_entry.items():
            if "demo" in key:
                attacked_context = value + prompt
                new_key = f"replaced_{key}"
                data_dict[new_key] = attacked_context
        
        dataset_list.append(data_dict)
    
    return dataset_list
        
def context_ingore(dataset_path: str):
    """"""

    prompt = context_ignore_prompt()
    
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = get_common_compression_dataset(dataset=dataset)

    dataset_list = []
    for data_entry in dataset:
        data_dict = data_entry.copy()
        for key, value in data_entry.items():
            if "demo" in key:
                attacked_context = value + prompt
                new_key = f"replaced_{key}"
                data_dict[new_key] = attacked_context
        
        dataset_list.append(data_dict)
    
    return dataset_list

def fake_completion():
    """"""

