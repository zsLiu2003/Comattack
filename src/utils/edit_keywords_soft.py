import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import csv
from tqdm import tqdm
from src.utils.get_prompt import get_edit_keywords_increase_prompt, get_edit_keywords_decrease_prompt
from src.data.data_process import get_common_compression_dataset, get_compression_dataset
import sys
def edit_keywords(model_path,output_path, dataset_path, keywords_dataset_path, flag="increase"):
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    output_path = f"{output_path}/keywords_{flag}.json"
    if flag == "increase":
        prompt = get_edit_keywords_increase_prompt()
    else:
        prompt = get_edit_keywords_decrease_prompt()

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = get_common_compression_dataset(dataset=dataset)
    keywords_dataset = load_dataset("json", data_files=keywords_dataset_path, split="train")

    edited_data = []
    for data, keywrods in tqdm(zip(dataset, keywords_dataset), desc="------------------edit the keywords with Qwen3-------------------"):
        user_input = ""
        question = data["question"]
        requirements = data["requirements"]
        user_input += f"'Question + Requirements': {question} + {requirements}\n"
        demo_edit = {}
        for i in range(len(keywrods)):
            demo = data[f"demo_{i+1}"]
            keyword = keywrods[f"demo{i+1}"]
            user_input += f"'Demo': {demo} \n 'Keywords: {keyword}'"
            messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
            ]

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
            demo_edit[f"demo_{i+1}"] = content
        
        edited_data.append(demo_edit)
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(edited_data, file, indent=4)
    
    print(f"------------------------------Successfully edit the keywords with {flag}!---------------------------------")

if __name__ == "__main__":
    
    flag = sys.argv[1]
    model_name = "models/Qwen3-32B"
    output_path = "src/data"
    dataset_path = "src/data/data.json"
    keywords_path = "src/data/data_keywords_with_Qwen3.json"

    edit_keywords(
        model_path=model_name,
        output_path=output_path,
        dataset_path=dataset_path,
        keywords_dataset_path=keywords_path,
        flag=flag,
    )
