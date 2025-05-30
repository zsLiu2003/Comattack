import torch
import transformers
from datasets import load_dataset, Dataset

def get_compression_dataset(dataset: None):

    new_dataset = []
    for origin_data in dataset:
        new_data = {}
        i = 0
        for key, value in origin_data.items():
            if i == 0:
                new_data["question"] = value
            elif i == 1:
                requirements = value
                requirements = requirements.split("; ")
                for j, requirement in enumerate(requirements):
                    new_data[f"requirement_{j+1}"] = requirement
            else:
                k = str(key[6])
                if int(k) == 6:
                    break
                new_data[f"demo_{k}"] = value
            i += 1
        
        new_dataset.append(new_data)

    return Dataset.from_list(new_dataset)
                 

def get_common_compression_dataset(dataset: None):

    new_dataset = []
    for data in dataset:
        new_data = {}
        for key, value in data.items():
            if "output" in key:
                k = str(key[6])
                if k==6: 
                    break
                new_data[f"demo_{k}"] = value
            else:
                new_data[key] = value
            
        new_dataset.append(new_data)
    
    return Dataset.from_list(new_dataset)

    

