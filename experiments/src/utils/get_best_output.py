# the following code is to get the best demo among the given 5 demos
# the demos here can be product description, agent tool description, and website html data


import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
from datasets import load_dataset,Dataset
from src.data.data_process import CompressionDataset,CompressionCommonDataset
from transformers import pipelines
import accelerate
import deepspeed


def get_parser():
    parser = argparse.ArgumentParser(description="The config of get best output")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/opt/model/models/Qwen3-32B",
        help="path of the LLM"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/lzs/compressionattack/experiments/src/data/data.json",
        help="path of the dataset",
    )
    
    parser.add_argument(
        "--compression_model_path",
        type=str,
        default="/opt/model/models/gpt2-dolly",
        help="path of compression model",
    )

    return parser

import random
import json
from tqdm import tqdm
from llmlingua import PromptCompressor

def get_best_output(
        other_dataset=None, 
        data_with_target_path="/home/lzs/compressionattack/experiments/src/data/data_with_target.json"
        ):
    
    from src.utils.get_prompt import get_target_prompt

    parser = get_parser()
    args = parser.parse_args()

    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    if other_dataset is not None:
        dataset = other_dataset     
    else:
        dataset = load_dataset("json", data_files=args.data_path, split="train")
        dataset = CompressionCommonDataset(dataset=dataset)
    # print(len(dataset))
    # model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map = device)
    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=args.model_path,
    #     model_kwargs={"torch_dtype": torch.bfloat16}
    #     device_map=device,
    # )
    # compression_model = AutoModelForCausalLM.from_pretrained(
    #     args.compression_model_path,
    #     device_map="cuda:0",
    #     torch_dtype=torch.bfloat16,
    # )
    # llmlingua_model = PromptCompressor(
    #     model_name=args.compression_model_name,
    #     device_map="cuda:6"
    # )

    model = AutoModelForCausalLM.from_pretrained(args.model_path,torch_dtype=torch.bfloat16,device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompt = get_target_prompt()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    new_dataset = []
    for data in tqdm(dataset):
        user_input = ""
        keys = []
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
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors='pt',
            return_attention_mask=True,
        ).to(model.device)
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id = terminators,
            do_sample = True,
            temperature = 0.6,
            top_p = 0.9,
            pad_token_id = tokenizer.eos_token_id,
        )
        output_token_ids = outputs[0][input_ids.shape[-1]:]
        output = tokenizer.decode(output_token_ids, skip_special_tokens=True)
        # print(output)
        data["best"] = str(output)
        # keys = data.keys()
        remaining_keys = [k for k in keys if k != str(output)]
        data["target"] = str(random.choice(remaining_keys))
        
        new_dataset.append(data)
        
        with open(data_with_target_path, "w", encoding="utf-8") as file:
            json.dump(new_dataset, file, indent=4)
        
    return Dataset.from_list(new_dataset)


if __name__ == "__main__":
    
    get_best_output()