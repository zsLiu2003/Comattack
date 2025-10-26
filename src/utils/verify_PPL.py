# this code is to verify that the text with a higher PPL will be recommanded by LLM
# or verify others: what determines the quality of text which will be selected by LLM? This is the first question.
# this code has verfied that the model will select the demo with a higher mean PPL.
# so the PPL of the demo is also a part of the recommendation

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmlingua import PromptCompressor
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import pipeline
import argparse
import numpy as np


class CompressDataset(Dataset):
    def __init__(self, dataset, tokenizer) -> None:
        super().__init__()
        self.data = dataset 
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        original_data = self.data[index]
        # encode_function = lambda x: self.tokenizer(x, return_tensors='pt')
        # new_data = {
        #     "instruction": original_data["instruction"],
        #     "output1": original_data["text1"],
        #     "output2": original_data["text2"],
        #     "target": original_data["target"]
        #      }
        new_data = {}
        i = 0
        for k, v in original_data:
            if i == 0:
                new_data["instrucation"] = v
            elif i == len(original_data):
                new_data["target"] = v
            else:
                key = str(f"output_{i}")
                new_data[key] = v
            i+=1

        # new_data = {k: self.encode(v, return_tensors='pt') for k,v in original_data}
        return new_data    

def get_sequence_length_and_output(tokenizer, text):

    le_list = []
    output_list = []
    for k ,v in text:
        if "output" in k:
            le_list.append(tokenizer(v, return_tensors='pt')["input_ids"].size(-1))
            output_list.append(v)
    # return [input_ids1.size(-1), input_ids2.size(-1)]
    return le_list, output_list

def compress_text(examples, llmlingua_model=None, tokenizer=None):

    le, output = get_sequence_length_and_output(
        tokenizer=tokenizer,
        text=examples,
    )
    
    return [llmlingua_model.compress_prompt(
        output[i],
        target_token=le[i]
    ) for i in range(len(le)) ]

def get_topk(ppl_list, input_ids, top_k, tokenizer):

    top_k = min(top_k, len(ppl_list))
    topk_vals, topk_ids_index = torch.topk(ppl_list, k=top_k, largest=True)
    topk_vals = topk_vals.tolist()
    # print(type(topk_vals))
    token_ids = input_ids.squeeze(0)[1:]
    topk_token_ids = token_ids[topk_ids_index]
    topk_tokens = tokenizer.convert_ids_to_tokens(topk_token_ids.tolist())
    token_list_with_ppl = [
        {"token": token, "ppl": ppl}
        for token, ppl in zip(topk_tokens, topk_vals)
    ]

    return token_list_with_ppl


def get_bottomk(ppl_list, input_ids, top_k, tokenizer):

    bottom_k = min(top_k, len(ppl_list))
    bottomk_vals, bottomk_ids_index = torch.topk(ppl_list, k=bottom_k, largest=False)
    bottomk_vals = bottomk_vals.tolist()
    token_ids = input_ids.squeeze(0)[1:]
    bottom_token_ids = token_ids[bottomk_ids_index]
    bottomk_tokens = tokenizer.convert_ids_to_tokens(bottomk_ids_index.tolist())
    token_list_with_ppl = [
        {"token": token, "ppl": ppl}
        for token, ppl in zip(bottomk_tokens, bottomk_vals)
    ]

    return token_list_with_ppl

# get the ppl of every token and the mean PPL of sentence
def get_PPL(model, tokenizer, origin_text, compressed_text, top_k):

    device = model.device
    input = tokenizer(origin_text, return_tensors='pt').to(device)
    input_ids1 = input["input_ids"]
    input_ids2 = tokenizer(compressed_text, return_tensor='pt')["input_ids"]
    
     # get the logits 
    with torch.no_grad():
        output = model(**input)
        logits = output.logits

    # calculate the cross_entropy loss
    shift_logits = logits[...,:-1, :].contiguous()
    label_logits = input_ids1[...,1:].contiguous()
    loss_function = CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
    loss = loss_function(
        shift_logits.view(-1,shift_logits.size(-1)), 
        label_logits.view(-1),
        )
    ppl_per_token = torch.exp(loss).cpu().numpy().tolist()
    ppl_mean_origin = torch.exp(loss).mean().item()

    # select the ppl of compressed_tokens
    le = min(input_ids1.size(1), len(ppl_per_token))
    le2 = input_ids2.size(1)
    id2_index = 0
    compressed_ppl_per_token = []
    for id1_index in range(le):
        if id2_index < le2 and input_ids1[0, id1_index] == input_ids2[0, id2_index]:
            compressed_ppl_per_token.append(ppl_per_token[id1_index])
            id2_index += 1
    ppl_mean_compressed = np.mean(compressed_ppl_per_token)
    
    # ppl_per_token.sort(reverse=True)
    # compressed_ppl_per_token.sort(reverse=True)
    result_token_list_origin = get_topk(
        ppl_list=ppl_per_token,
        input_ids=input_ids1,
        top_k=top_k,
        tokenizer=tokenizer,
        )
    return  result_token_list_origin, ppl_mean_origin, ppl_mean_compressed


# get the ppl of the single text, get the top_k ppl tokens
def get_single_PPL(model, tokenizer, text, top_k, flag: bool):

    device = model.device
    input = tokenizer(text, return_tensors="pt").to(device)
    input_ids = input["input_ids"] # shape = [batch_size, sequence_length-1]

    with torch.no_grad():
        output = model(**input)
        logits = output.logits # shape = [batch_size, sequence_length/num_steps, vocabulary_size]

    shift_logits = logits[..., :-1, :].contiguous() # shape = [batch_size, sequence_length-1, vocabulary_size] 
    label_logits = input_ids[:,1:].contiguous() # shape = [batch_size, sequence_length-1]
    loss_function = CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
    loss = loss_function(
        shift_logits.view(-1, shift_logits.size(-1)),
        label_logits.view(-1)
    )
    # ppl_per_token = torch.exp(loss).float().cpu().numpy().tolist()
    ppl_per_token = torch.exp(loss)
    ppl_mean_origin = torch.exp(loss).mean().item()
    
    # type: list
    top_k_tokens_with_ppl_list= get_topk(
        ppl_list=ppl_per_token,
        input_ids=input_ids,
        top_k=top_k,
        tokenizer=tokenizer
    )
    if not flag:
        bottom_k_tokens_with_ppl_list = get_topk(
            ppl_list=ppl_per_token,
            input_ids=input_ids,
            top_k=top_k,
            tokenizer=tokenizer,
        )
        return top_k_tokens_with_ppl_list, bottom_k_tokens_with_ppl_list, ppl_mean_origin
    
    return top_k_tokens_with_ppl_list, ppl_mean_origin



# the following code is to get the recommandation output of LLM
def large_model_output(model, tokenizer, text):
    """"""
    
# get the config of verify_PPL
def get_parser():
    parser = argparse.ArgumentParser(description="The config of Verify_PPL:")

    parser.add_argument(
        "--compression_model_name",
        type=str,
        default="models/gpt2-dolly",
        help="name of compression_model"
    )
    parser.add_argument(
        "--large_model_name",
        type=str,
        default="models/Llama-2-7b-chat-hf",
        help="name of Large Model",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="models/dataset/llmbar_train_1.csv",
        help="path of the dataset",
    )
    
    return parser


def main():
    
    parser = get_parser()
    args = parser.parse_args()

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    device2 = "cuda:2" if torch.cuda.is_available() else "cpu"

    dataset = load_dataset("csv", data_files=args.data_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.compression_model_name)
    
    # use the eos_token as the pad_token, because the default pad_token in some model's tokenzier is None.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    compression_model = AutoModelForCausalLM.from_pretrained(
        args.compression_model_name,
        device_map=device,
        torch_dtype=torch.float16) 
    llmlingua_model = PromptCompressor(model_name=args.compression_model_name, device_map=device)
    
    original_dataset = CompressDataset(dataset=dataset, tokenizer=tokenizer)
    compressed_dataset = original_dataset.map(compress_text, fn_kwargs={
        "llmlingua_model": llmlingua_model,
        "tokenizer": tokenizer,
    },
    )
    dataloader = DataLoader(dataset=compressed_dataset, batch_size=1, shuffle=True)
    ppl = {}
    ppl["target"] = []
    ppl["others"] = []

    for batch in dataloader:
        for compressed_prompt in batch:        
           
           ppl.append([get_PPL(
               model=compression_model,
               tokenizer=tokenizer,
               text=compressed_prompt["compressed_prompt"],
               device=device,
               top_k=10,
           )]) 
    
if __name__ == "__main__":

    main()