from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import csv
import os
from tqdm import tqdm
from datasets import load_dataset
from src.data.data_process import get_compression_dataset,get_common_compression_dataset
from src.utils.get_prompt import get_keywords_prompt
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import argparse
import torch
import re
# from src.util

# def get_parser():

#     parser = argparse.ArgumentParser(description="The config of get keywords from demos")
    
#     parser.add_argument(
#         "--model_name",
#         type=str,
#         default="models/Qwen3-32B",
#         help="name of extraction model",
#     )

#     parser.add_argument(
#         "--data_path",
#         type=str,
#         default="./src/data/data.json",
#         help="path of dataset with target"
#     )

#     parser.add_argument(
#         "--device",
#         default="cuda: 7" if torch.cuda.is_available() else "cpu",
#         help="device map",
#     )
    
#     # parser.add_argument(
#     #     "--prompt_path",
#     #     default="./src/data/get_keywords_prompt.txt",
#     #     type=str,
#     #     help="path of the prompt"
#     # )
#     return parser

def get_keywords(
    model_path,
    dataset,
    output_path="",
    device="cpu",
):

    # args = get_parser()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    prompt = get_keywords_prompt()
    
    new_dataset = []
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
        
        # print(content)
        json_pattern = re.compile(r"{.*?}", re.DOTALL)
        json_match = json_pattern.search(str(content))
        if json_match:
            extracted_json_string = json_match.group(0)
        # print(repr(extracted_json_string))
        new_data = json.loads(extracted_json_string)
        # print(new_data)
        final_dict = {}

        for key, value in new_data.items():
            if isinstance(value, str):
                try:
                    final_dict[key] = json.loads(value)
                except json.JSONDecodeError:
                    final_dict[key] = value
            else:
                
                final_dict[key] = value
        print(final_dict)
        new_dataset.append(final_dict)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(new_dataset, file, indent=4)

if __name__ == "__main__":

    dataset = load_dataset("json", data_files="src/data/data.json", split="train")
    dataset = get_common_compression_dataset(dataset=dataset)
    dataset = dataset.select(range(50))
    get_keywords(
        model_path="models/Qwen3-32B",
        dataset=dataset,
        output_path="src/data/revised_keywords_with_Qwen3_1.json",
        device="auto",
    )
