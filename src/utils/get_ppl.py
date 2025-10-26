from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
import torch
import csv
import json
from tqdm import tqdm
from datasets import load_dataset, Dataset
import argparse
from src.data.data_process import get_common_compression_dataset
from src.utils.get_compressed_text import get_compressed_text
# import deepspeeds
from src.utils.verify_PPL import get_PPL, get_single_PPL
from src.utils.get_best_output import get_best_output
import accelerate
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# def get_parser():

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--compression_model_path",
#         type=str,
#         default="models/gpt2-dolly",
#         help="path of model to calculte the PPL of every token"
#     )

#     parser.add_argument(
#         "--model_path",
#         type=str,
#         default="models/Llama-2-7b-chat-hf",
#     )

#     parser.add_argument(
#         "--dataset_path",
#         type=str,
#         default="./src/data/data.json",
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
       

def get_mean_PPL(
    # model_path: str,
    compression_model_path: str,
    dataset: Dataset,
    top_k: int,
    output_path: str="src/data",
    target_token: int=50,
):
    if "gpt2" in compression_model_path:
        ppl_model_name = "gpt2"
    elif "Llama" in compression_model_path:
        ppl_model_name = "llama2"
    elif "phi" in compression_model_path:
        ppl_model_name = "phi"
    
    
    # if "Qwen" in model_path:
    #     model_name = "Qwen3"
    # elif "Llama" in model_path:
    #     model_name = "Llama2"

    model = AutoModelForCausalLM.from_pretrained(compression_model_path,torch_dtype=torch.bfloat16,device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(compression_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # assert len(dataset) == len(compressed_dataset)
    ppl_result = []
    
    dataset = get_common_compression_dataset(dataset=dataset)
    for data in tqdm(dataset, desc="get the PPL of every demo in each data entry."):
        ppl_dict = {}
        for key, value in data.items():
            if "demo" in key:
                high_ppl_tokens, mean_PPL = get_single_PPL(
                    model=model,
                    tokenizer=tokenizer,
                    text=value,
                    top_k=top_k,
                    flag=True,
                )
                ppl_dict[key] = mean_PPL
        ppl_result.append(ppl_dict)

    
    output_path = f"{output_path}/mean_ppl_origin_{ppl_model_name}.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(ppl_result, file, indent=4)
    
    print("------------------Successfully finished the calculation of mean PPL of every demo.--------------------")

    
        
def get_ppl(
    model_path: str,
    compression_model_path: str,
    dataset: Dataset,
    top_k: int,
    output_path: str="src/data",
    target_token: int=50,
):
    
    if "gpt2" in compression_model_path:
        ppl_model_name = "gpt2"
    elif "Llama" in compression_model_path:
        ppl_model_name = "llama2"
    elif "phi" in compression_model_path:
        ppl_model_name = "phi"

    # parser = get_parser()
    # args = parser.parse_args()
    
    # dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = get_common_compression_dataset(dataset=dataset)
    compressed_dataset = get_compressed_text(
        model_name=compression_model_path,
        dataset=dataset,
        device="cuda:0",
        target_token=target_token,
        output_path=f"{output_path}/compressed_data_{ppl_model_name}.json",
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
    
    """
    the following code is used to get the best output
    """
    # dataset = get_best_output(
    #     model_path=model_path,
    #     compression_model_path=compression_model_path,
    #     other_dataset=dataset,
    #     data_with_target_path=f"{output_path}/data_with_target_{model_name}.json"
    # )
    # compressed_dataset = get_best_output(
    #     model_path=model_path,
    #     compression_model_path=compression_model_path,
    #     other_dataset=compressed_dataset,
    #     data_with_target_path=f"{output_path}/data_with_compressed_target_{model_name}.json"
    #     )
    
    """
    the following code is to load the model into GPUs
    """
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
    
    
    model = AutoModelForCausalLM.from_pretrained(compression_model_path,torch_dtype=torch.bfloat16,device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(compression_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    assert len(dataset) == len(compressed_dataset)
    ppl_result = []

    """
    for data in tqdm(dataset, desc="Get the PPL of every token: "):
        # assert len(data) == len(compressed_data), "the length of original data is not equal to the compressed data"
        ppl_datadict = {}
        # for original_demo, compressed_demo in :
        #     original_key, original_value = original_demo
        #     compressed_key, compressed_value = compressed_demo
            
        #     token_list, ppl_mean_origin, ppl_mean_compressed = get_PPL(
        #         model=model,
        #         tokenizer=tokenizer,
        #         origin_text=original_demo,
        #         compressed_text=compressed_demo,
        #         top_k=top_k,
        #     )
        #     ppl_data = {
        #         "token_list": token_list,
        #         "ppl_mean_origin": ppl_mean_origin,
        #         "ppl_mean_compressed": ppl_mean_compressed, 
        #     }
        #     ppl_datadict[original_key] = ppl_data
        original_demo, compressed_demo = data["all_demos"], data["compressed"]
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
        ppl_result.append(ppl_data)
    """
    for data in tqdm(compressed_dataset, desc="get the top_k tokens with high ppl of every demo"):
        ppl_dict = {}
        for key, value in data.items():
            if "demo" in key:
                topk_tokens_with_ppl, mean_ppl = get_single_PPL(
                    model=model,
                    tokenizer=tokenizer,
                    text=value,
                    top_k=top_k,
                    flag=True,
                )
                ppl_dict[key] = {
                    "tokens_with_ppl": topk_tokens_with_ppl,
                    "mean_ppl": mean_ppl,
                }
        ppl_result.append(ppl_dict)

    ppl_dataset_path = f"{output_path}/ppl_data_{ppl_model_name}.json"
    with open(ppl_dataset_path, "w", encoding="utf-8") as file:
        json.dump(ppl_result, file, indent=4)
    
    mean_ppl_result = []
    for data in tqdm(compressed_dataset, desc="get the mean ppl of origin_demo and compressed demo"):
        origin_text = data["original_text"]
        compressed_text = data["compressed_text"]
        _, original_mean_ppl = get_single_PPL(
            model=model,
            tokenizer=tokenizer,
            text=origin_text,
            top_k=top_k,
            flag=True,
        )
        _, compressed_mean_ppl = get_single_PPL(
            model=model,
            tokenizer=tokenizer,
            text=compressed_text,
            top_k=top_k,
            flag=True,
        )
        mean_ppl_result.append(
            {
                "original_mean_ppl": original_mean_ppl,
                "compressed_mean_ppl": compressed_mean_ppl,
            }
        )

    mean_ppl_dataset_path = f"{output_path}/mean_ppl_{ppl_model_name}.json"    
    with open(mean_ppl_dataset_path, "w", encoding="utf-8") as file:
        json.dump(mean_ppl_result, file, indent=4)
    
    print("------------Successfully get the PPL of every token and the whole demo!-----------")


    
if __name__ == "__main__":
    
    import sys
    compression_model_name = sys.argv[1]

    dataset = load_dataset("json", data_files="src/data/data.json", split="train")

    # get_ppl(
    #     model_path="models/Qwen3-32B",
    #     compression_model_path=str(compression_model_name),
    #     dataset=dataset,
    #     top_k=20,
    #     output_path="src/data",
    #     target_token=100,
    # )
    get_mean_PPL(
        model_path="models/Qwen3-32B",
        compression_model_path=str(compression_model_name),
        dataset=dataset,
        top_k=20,
        output_path="src/data",
        target_token=100,
    )