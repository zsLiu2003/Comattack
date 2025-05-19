from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
import torch
import csv
import json
from tqdm import tqdm
from datasets import load_dataset
import argparse
from src.data.data_process import CompressionCommonDataset
from src.utils.get_compressed_text import get_compressed_text
import deepspeed
from src.utils.verify_PPL import get_PPL
from src.utils.get_best_output import get_best_output
import accelerate
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# def get_parser():

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--compression_model_path",
#         type=str,
#         default="/opt/lzs/models/gpt2-dolly",
#         help="path of model to calculte the PPL of every token"
#     )

#     parser.add_argument(
#         "--model_path",
#         type=str,
#         default="/opt/model/models/Llama-2-7b-chat-hf",
#     )

#     parser.add_argument(
#         "--dataset_path",
#         type=str,
#         default="/home/lzs/compressionattack/experiments/src/data/data.json",
#     )

#     parser.add_argument(
#         "--top_k",
#         type=int,
#         default=20,
#         help="the number of selected tokens with a high PPL"
#     )

#     return parser

# def pure_dataset(examples):
    
#     for key,value in examples.items():
        
# def replace_demos(example, idx, compressed_dataset, demo_keys):
    
#     compressed_prompt = compressed_dataset[idx]
#     for key in demo_keys:
#         example[key] = compressed_prompt[key]
    
#     return example
       

def get_ppl(
    model_path,
    compression_model_path,
    dataset,
    top_k,
    output_path="/home/lzs/compressionattack/experiments/src/data",
):

    # parser = get_parser()
    # args = parser.parse_args()
    
    # dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = CompressionCommonDataset(dataset=dataset)
    compressed_dataset = get_compressed_text(
        model_name=compression_model_path,
        dataset=dataset,
        device="cuda:0",
    )
    # demo_keys = [key for key in dataset.keys() if "demo" in key]
    # dataset.map(
    #     replace_demos,
    #     with_indics=True,
    #     batched=False,
    #     fn_kwargs={
    #         "compressed_dataset": compressed_dataset,
    #         "demo_keys": demo_keys, 
    #     }
    # )
    if "Qwen" in model_path:
        model_name = "Qwen3"
    elif "Llama" in model_path:
        model_name = "Llama2"

    # output_data_path = f"{output_path}/data_with_compressed.json"
    # with open(output_data_path, "w", encoding="utf-8") as file:
    #     json.dump(dataset, file, indent=4)
    
    dataset = get_best_output(
        other_dataset=dataset,
        data_with_target_path=f"{output_path}/data_with_target.json_{model_name}"
    )
    compressed_dataset = get_best_output(
        other_dataset=compressed_dataset,
        data_with_target_path=f"{output_path}/data_with_compressed_target.json_{model_name}"
        )
    
    # load the Qwen3-32 model with two L40s
    # if model_name == "Qwen3":
    #     max_memory = {
    #         0: "45GB",
    #         1: "45GB",
    #     }
    #     with init_empty_weights():
    #         model = AutoModelForCausalLM.from_pretrained(
    #             args.large_model,
    #             torch_dtype=torch.bfloat16,
    #             trust_remote_code=True,
    #             device_map="auto",
    #             low_cpu_mem_usage=True,
    #         )
    #         # config = AutoConfig.from_pretrained(args.large_model, trust_remote_code=True)
    #     model = load_checkpoint_and_dispatch(
    #         model,
    #         args.large_model,
    #         device_map="auto",
    #         offload_folder=None,
    #         dtype=torch.bfloat16,
    #         no_split_module_classes=["GPTQEmbedding"],
    #         )
        
    #     print(f"----------------The layer distribution of {args.large_model}-------------------")
    #     for name, device in model.hf_device_map.items():
    #         print(f"{name}:{device}")
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(args.large_model, torch_dtype=torch.bfloat16, device_map='cuda:7')

    # remain_columns = [key for key in dataset.keys() if "demo" in key]
    # remain_dataset = dataset.select_columns(remain_columns)
    model = AutoModelForCausalLM.from_pretrained(compression_model_path,device_map="cuda:7")
    tokenizer = AutoTokenizer.from_pretrained(compression_model_path)
    assert len(dataset) == len(compressed_dataset)
    ppl_result = []
    for data, compressed_data in tqdm(zip(dataset, compressed_dataset), desc="Get the PPL of every token: "):
        assert len(data) == len(compressed_data), "the length of original data is not equal to the compressed data"
        ppl_datadict = {}
        for original_demo, compressed_demo in zip(data[-7:-2], compressed_data[-7:-2]):
            original_key, original_value = original_demo
            compressed_key, compressed_value = compressed_demo
            
            token_list, ppl_mean_origin, ppl_mean_compressed = get_PPL(
                model=model,
                tokenizer=tokenizer,
                origin_text=original_demo,
                compressed_text=compressed_demo,
                top_k=top_k,
            )
            ppl_data = {
                "token_list": token_list,
                "ppl_mean_origin": ppl_mean_origin,
                "ppl_mean_compressed": ppl_mean_compressed, 
            }
            ppl_datadict[original_key] = ppl_data
        
        ppl_result.append(ppl_datadict)
    
    ppl_dataset_path = f"{output_path}/ppl_data.json"
    
    with open(ppl_dataset_path, "w", encoding="utf-8") as file:
        json.dump(ppl_result, file, indent=4)
    
    

            
            
            
            
    
    