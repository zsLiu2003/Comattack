# this script is used to filter out long data entries from the dataset.
# for llmlingua1, the used model is phi2 and and NousResearch/Llama-2-7b-hf
# the max_token_length is 2k for phi2 and 4k for NousResearch/Llama-2-7b-hf
# for llmlingua2, the used model are two bert models, microsoft/llmlingua-2-xlm-roberta-large-meetingbank and a smaller one, microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
# the max_token_length is 
# In our setting, we set the max_token_length to 2k for llmlingua1 and 512 for llmlingua2, therefore we need to filter out the data entries that have more than 2k tokens for llmlingua1 and more than 512 tokens for llmlingua2.

"""
The token sequence length of system prompts + query + suffix should be less than the max_token_length
"""

from datasets import load_dataset
import json
from transformers import AutoTokenizer

def filter_long_data_entries(dataset_path: str, output_path: str, method: str, model_name: str):
    
    if method == "llmlingua1":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        max_token_length = 1800 # We leave a margin of 200 tokens for the suffix
    elif method == "llmlingua2":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        max_token_length = 480 # We leave a margin of 32 tokens for the suffix
    else:
        raise ValueError(f"Invalid method: {method}")
    
    dataset = json.load(open(dataset_path, "r", encoding="utf-8"))
    filtered_dataset = []
    total_filtered_entries = 0
    original_entries = 0
    for item in dataset:
        system_prompt = item["system_prompt"]
        token_sequence = tokenizer.encode(system_prompt, add_special_tokens=False)
        original_entries += 1
        if len(token_sequence) > max_token_length:
            continue
        total_filtered_entries += 1
        filtered_dataset.append(item)
    # for item in dataset:
    #     system_prompt = item["system_prompt"]
    #     guardrail_list = item["guardrail_list"]
    #     new_guardrail_list = []
    #     for guardrail_dict in guardrail_list:
    #         adversarial_query = guardrail_dict["adversarial_query"]
    #         prompt = "System Prompt: " + system_prompt + "\n\n" + "Question: " + adversarial_query
    #         token_sequence = tokenizer.encode(prompt, add_special_tokens=False)
    #         original_entries += 1
    #         if len(token_sequence) > max_token_length:
    #             continue
    #         total_filtered_entries += 1
    #         guardrail_dict["entire_prompt"] = prompt
    #         new_guardrail_list.append(guardrail_dict)
        
    #     if len(new_guardrail_list) == 0:
    #         continue
    #     item["guardrail_list"] = new_guardrail_list
    #     filtered_dataset.append(item)

    json.dump(filtered_dataset, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    print(f"Filtered {len(filtered_dataset)} data entries from {dataset_path} to {output_path}")
    print(f"The original dataset has {original_entries} data entries")
    print(f"The filtered dataset has {total_filtered_entries} data entries")
    # print(f"The ratio of the filtered dataset to the original dataset is {total_filtered_entries / original_entries}")

if __name__ == "__main__":
    # dataset_path = "data/guardrail_violation_queries_for_leaked_system_prompt.json"
    dataset_path = "data/extracted_guardrails_reduced_filtered.json"
    output_path = "data/guardrail_violation_queries_for_leaked_system_prompt_filtered_llmlingua1_v1.json"
    method = "llmlingua1"
    model_name = "microsoft/phi-2" # "microsoft/phi-2"
    # model_name = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank" # "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    filter_long_data_entries(dataset_path, output_path, method, model_name)


