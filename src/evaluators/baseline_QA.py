from datasets import load_dataset
import re  
from src.data.data_process import get_QA_dataset
import json
import random
from src.attacks.baseline_prompt_QA import context_ignore_prompt,fake_completion_prompt,direct_injection_prompt,escape_characters_prompt


def clean(dataset_path):

    dataset = get_QA_dataset(
        dataset_path=dataset_path
    )

    return dataset

def naive_attack(dataset_path):
    """"""
    """
    Randomly replaces ~10 characters in the context field for QA tasks.
    """
    dataset = get_QA_dataset(
        dataset_path=dataset_path,
    )

    dataset_list = []

    for sample in dataset:
        context = sample["context"]
        context_chars = list(context)
        indices = random.sample(range(len(context_chars)), min(10, len(context_chars)))

        for idx in indices:
            old_char = context_chars[idx]
            if old_char.isalpha():
                context_chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz'.replace(old_char.lower(), ''))
            else:
                context_chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')

        attacked_context = ''.join(context_chars)

        new_sample = sample.copy()
        new_sample["context"] = attacked_context
        dataset_list.append(new_sample)

    return dataset_list
    
    

def direct_injection(dataset_path: str):
    """"""
    
    prompt = direct_injection_prompt()
    dataset = get_QA_dataset(
        dataset_path=dataset_path,
    )
    dataset_list = []
    for data_entry in dataset:
        context = data_entry["context"] + prompt
        new_data_entry = data_entry.copy()
        
        new_data_entry["context"] = context
        dataset_list.append(new_data_entry)
    
    return dataset_list    


def escape_characters(dataset_path: str):
    """"""
    prompt = escape_characters_prompt()
    dataset = get_QA_dataset(
        dataset_path=dataset_path,
    )
    dataset_list = []
    for data_entry in dataset:
        context = data_entry["context"] + prompt
        new_data_entry = data_entry.copy()
        
        new_data_entry["context"] = context
        dataset_list.append(new_data_entry)
    
    return dataset_list    

def context_ingore(dataset_path: str):
    """"""
    prompt = context_ignore_prompt()
    dataset = get_QA_dataset(
        dataset_path=dataset_path,
    )
    dataset_list = []
    for data_entry in dataset:
        context = data_entry["context"] + prompt
        new_data_entry = data_entry.copy()
        
        new_data_entry["context"] = context
        dataset_list.append(new_data_entry)
    
    return dataset_list   

def fake_completion():
    """"""

