from transformers import AutoModelForCausalLM, AutoTokenizer
from llmlingua import PromptCompressor
from datasets import load_dataset
from src.data.data_process import get_QA_dataset
from src.utils.get_prompt import get_QA_prompt

import torch

import json
from tqdm import tqdm


def QA_test_result(context_dataset_path, question_dataset_path, compressed_model_name, large_model_name, key, output_path, device: str="cuda:0"):
    
    context_dataset = load_dataset("json", data_files=context_dataset_path, split="train")
    question_dataset = get_QA_dataset(
        dataset_path=question_dataset_path
    )
    print(f"-----------Information of the dataset: {context_dataset}------------")
    
    large_model = AutoModelForCausalLM.from_pretrained(large_model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(large_model_name)
    
    compression_model = PromptCompressor(
        model_name=compressed_model_name,
        device_map=device,
    )
    
    system_prompt = get_QA_prompt()

    output_list = []
    for data_entry in tqdm(context_dataset, desc="---------Execute the QA task---------"):
        output_dict = {}
        idx = data_entry["idx"]
        # original_context = data_entry["original"]
        # optimized_context = data_entry["optimized"]
        context = data_entry[key]
        question = question_dataset[idx-1]["question"]
        answers = question_dataset[idx-1]["answers"]["text"][0]

        print(f"-----Original length: {len(context)}-----")
        # execute the compression process
        if len(context) < 900:
            le = 25
        else:
            le = 50
        compressed_context = compression_model.compress_prompt(
            context,
            instruction="",
            question="",
            target_token=le,
        )
        
        compressed_context = compressed_context["compressed_prompt"]
        print(f"-----Compressed length: {len(compressed_context)}-----")
        # compressed_optimized_context = compression_model.compress_prompt(
        #     optimized_context,
        #     instruction="",
        #     question="",
        #     target_token=100,
        # )
        prompt = f"The context is: {compressed_context},\n the question is: {question},\n please answer my question"
        # optimized_prompt = f"The context is: {compressed_optimized_context},\n the question is: {question},\n please answer my question"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(large_model.device)
        generated_ids = large_model.generate(
            **model_inputs,
            max_new_tokens=32768,
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        output_dict["idx"] = idx
        output_dict[key] = context 
        output_dict["compressed_context"] = compressed_context
        output_dict["question"] = question
        output_dict["answer"] = content
        output_dict["real_answer"] = answers
        print(output_dict)

        output_list.append(output_dict)
        # break
    
    output_path = f"{output_path}/QA_{key}_results2.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output_list, file, indent=4)

if __name__ == "__main__":

    question_dataset_path = "squad/validation-00000-of-00001.parquet"
    context_dataset_path = "src/data/QA_keywords_edit.json"
    compressed_model_name = "models/gpt2-dolly"
    large_model_name = "models/Qwen3-32B"
    output_path = "src/data"

    keys = ["original","optimized"]
    
    for key in keys:
        QA_test_result(
            context_dataset_path=context_dataset_path,
            question_dataset_path=question_dataset_path,
            compressed_model_name=compressed_model_name,
            large_model_name=large_model_name,
            key=key,
            output_path=output_path,
        )
