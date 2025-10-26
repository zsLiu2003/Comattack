# calculate_clean_ppl.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
# from dataloader.dataloader_squad import 
from src.data.data_process import get_QA_dataset
import numpy as np
import sys
# Define the get_ppl function (removing 'self' since it's not in a class)
def get_ppl(text: str, model, tokenizer) -> float:
    """
    Calculates the overall perplexity of a given text.
    """
    device = model.device
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    
    max_length = model.config.n_positions
    if input_ids.size(1) > max_length:
        input_ids = input_ids[:, :max_length]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    ppl = torch.exp(loss)
    return ppl.item()

if __name__ == "__main__":
    # Load model and tokenizer from the first script
    model_name = "models/gpt2-large"
    model = GPT2LMHeadModel.from_pretrained(model_name).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load SQuAD dataset
    raw_dataset = load_dataset("parquet", data_files={
        "train": "./data/squad/train-00000-of-00001.parquet",
        "validation": "./data/squad/validation-00000-of-00001.parquet"
    })
    # val_dataset = SquadDataset(data=raw_dataset, split="validation")
    # val_loader = DataLoader(val_dataset, batch_size=1)
    # val_loader = load_dataset("json", data_files="src/data/QA_keywords_edit.json", split="train")
    # Calculate perplexity for each context
    # contexts = []
    # for i, sample in enumerate(val_loader):
    #     context = sample["optimized"]
    #     ppl = get_ppl(context, model, tokenizer)
    #     # print(f"Context {i}: {context[:50]}... Perplexity: {ppl:.2f}")
    #     contexts.append({
    #         'context': context,
    #         'ppl': ppl
    #     })
        
    # ppl_values = [item['ppl'] for item in contexts]  # Extract all ppl values
    # if ppl_values:  # Ensure the list is not empty
    #     ppl_threshold = np.percentile(ppl_values, 99)
    #     print(f"Perplexity threshold (99th percentile): {ppl_threshold:.2f}")

    # if ppl_values:  # Ensure the list is not empty
    #     ppl_threshold = 50.63
    #     print(f"Perplexity threshold (99th percentile): {ppl_threshold:.2f}")

    #     # Calculate proportion of samples exceeding the threshold
    #     num_exceed = sum(p > ppl_threshold for p in ppl_values)
    #     ratio_exceed = num_exceed / len(ppl_values)
    #     print(num_exceed)
    #     print(f"Proportion of contexts exceeding the PPL threshold: {ratio_exceed:.2%}")
    
    original_ppls = []
    replaced_ppls = []
    dataset_path_list = [
        "src/data/replaced_confused_recommendation.json",
        "src/data/replaced_ppl_adjective_increase.json",
        "src/data/replaced_ppl_connectors_decrease.json",
        "src/data/replaced_ppl_prep_context_decrease.json",
        "src/data/replaced_ppl_synonym_decrease.json",
        "src/data/replaced_ppl_synonym_increase.json",
    ]
    for dataset_path in dataset_path_list:
        val_loader = load_dataset("json", data_files=dataset_path, split="train")
        for i, sample in enumerate(val_loader):
            for key, value in sample.items():
                original = value["original"]
                replaced = value["replaced"]

                # 计算 PPL
                ppl_orig = get_ppl(original, model, tokenizer)
                ppl_repl = get_ppl(replaced, model, tokenizer)

                original_ppls.append(ppl_orig)
                replaced_ppls.append(ppl_repl)

        # 步骤 1 和 2：计算 original 的 99% 分位数
        if original_ppls:
            ppl_threshold = np.percentile(original_ppls, 99)
            print(f"99th percentile PPL of original prompts in {dataset_path}: {ppl_threshold:.2f}")

            # 步骤 3：计算 replaced 中超过该阈值的比例
            num_exceed = sum(p > ppl_threshold for p in replaced_ppls)
            ratio_exceed = num_exceed / len(replaced_ppls)
            print(f"Proportion of replaced prompts exceeding original 99th PPL in {dataset_path}: {ratio_exceed:.2%}")