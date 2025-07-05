from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from src.utils.get_prompt import get_tool_keywords_prompt
from tqdm import tqdm

def get_tool_selection_keywords(
    model_path,
    dataset,
    output_path="",
    device="cpu",
):
    # model = AutoModelForCausalLM.from_pretrained()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    prompt = get_tool_keywords_prompt()
    
    new_dataset = []
    for data in tqdm(dataset, desc="Process tool selection dataset"):
        user_input = ""
        for key, value in data.items():
            user_input += f"{key}: {value}"
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ]