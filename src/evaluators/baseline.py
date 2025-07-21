import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmlingua import PromptCompressor
from datasets import load_dataset
from src.data.data_process import get_QA_dataset

def clean(dataset_path):

    if "QA" in dataset_path:
        dataset = get_QA_dataset(dataset_path=dataset_path)
    else:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    return dataset

def naive_attack():
    """"""

def direct_injection():
    """"""

def escape_characters():
    """"""

def context_ingore():
    """"""

def fake_completion():
    """"""

