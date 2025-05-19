# this code is a easy try to the attack

from transformers import AutoModelForCausalLM 
from datasets import load_dataset
# dataset_path = "/opt/lzs/dataset/llmbar_train_1.csv"

# dataset = load_dataset("csv",data_files=dataset_path, split="train")
# # print(type(dataset[0]))
# for i, item in enumerate(dataset[0]):
#     print(i)
#     print()
# print(type(dataset))
# print(dataset[0])




# # model = AutoModelForCausalLM.from_pretrained("lgaalves/gpt2-dolly")

# prompt = ""
from src.data.data_process import CompressionDataset

dataset_path = "/home/lzs/compressionattack/experiments/src/data/data.json"
dataset = load_dataset("json", data_files=dataset_path, split="train")
dataset = CompressionDataset(dataset=dataset)
print(dataset[1])