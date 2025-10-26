from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import json
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
from src.utils.get_prompt import get_llama2_template, get_target_prompt, get_keywords_prompt
from src.data.data_process import get_common_compression_dataset
import random


def llama2_inference(model_name=None, dataset=None, device="cpu"):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.chat_template = get_llama2_template()

    if "7B" or "13B" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
    
def Qwen3_inference(model_name, dataset, device="cpu", output_path=None):
    """"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    prompt = get_target_prompt()
    new_dataset = []
    keys = [key for key in dataset.column_names if "demo" in key]
    for data in tqdm(dataset):
        user_input = ""
        # keys = []
        for key, value in data.items():
            if key == "question":
                user_input += f"{key}: {value}"
            else:
                user_input += f",\n{key}: {value}"
        # print(user_input)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ]
        
        # print(f"message = {messages}")
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors='pt',
            enable_thinking=True,
        )

        input_ids = tokenizer([text], return_tensors='pt').to(model.device)
        generate_ids = model.generate(
            **input_ids,
            max_new_tokens=32768,
        )

        output_ids = generate_ids[0][len(input_ids.input_ids[0]):].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True)
        data["best"] = str(content)
        remaining_keys = [key for key in keys if key != content]
        data["target"] = str(random.choice(remaining_keys))
        new_dataset.append(data)
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(new_dataset, file, indent=4)
            
        # print("thinking content:", thinking_content)
        # print("content:", content)    
def API_inference():
    """"""


if __name__ == "__main__":
    
    dataset = load_dataset("json", data_files="src/data/data.json", split="train")
    dataset = get_common_compression_dataset(dataset=dataset)

    Qwen3_inference(
        model_name="models/Qwen3-32B",
        dataset=dataset,
        device="auto",
        output_path="src/data/data_best_Qwen3.json"
    )
