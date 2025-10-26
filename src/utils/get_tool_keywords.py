from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from src.utils.get_prompt import get_tool_keywords_prompt
from src.data.data_process import process_tool_selection_dataset
from tqdm import tqdm
import re
import json

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

        text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
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

    output_path = f"{output_path}/tool_selection_keywords_feature_extraction.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(new_dataset, file, indent=4)

if __name__ == "__main__":

    model_name = "models/Qwen3-32B"
    # dataset_path = "./data/squad/validation-00000-of-00001.parquet"
    output_path = "src/data"

    dataset_path = "src/data/Feature_Extraction_tool_short.json"
    dataset = process_tool_selection_dataset(
        dataset_path=dataset_path
    )
    get_tool_selection_keywords(
        model_path=model_name,
        output_path=output_path,
        dataset=dataset,
    )
