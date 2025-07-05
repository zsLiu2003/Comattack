import torch
import transformers
from datasets import load_dataset, Dataset, concatenate_datasets
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_compression_dataset(dataset: None):

    """
    Get the data with the integrated requirements
    """
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
                # if int(k) == 6:
                #     break
                new_data[f"demo_{k}"] = value
            i += 1
        
        new_dataset.append(new_data)

    return Dataset.from_list(new_dataset)
                 

def get_pure_demo_dataset(dataset: None):

    new_dataset = []
    for data in dataset:
        new_data = {}
        for key, value in data.items():
            if "output" in key:
                k = str(key[6])
                new_data[f"demo_{k}"] = value
            
        new_dataset.append(new_data)
    
    return Dataset.from_list(new_dataset)

def get_common_compression_dataset(dataset: None):

    """
    Get the recommendation dataset with seperated requirements
    """

    new_dataset = []
    for data in dataset:
        new_data = {}
        for key, value in data.items():
            if "output" in key:
                k = str(key[6])
                new_data[f"demo_{k}"] = value
            else:
                new_data[key] = value
            
        new_dataset.append(new_data)
    
    return Dataset.from_list(new_dataset)

def get_tool_selection_dataset(extraction_domain, dataset_path, output_path):

    """
    Extract the tool description and API-name from the complicated dataset.
    """

    # extracted_domain = ["Feature Extraction", "Text-to-Image"]
    with open(dataset_path, "r", encoding='utf-8') as file:
        output_data = []
        for line in file:
            output_dict = {}
            data_entry = json.loads(line)
            if data_entry.get('api_data', {}).get("functionality") == extraction_domain:
                output_dict["api_name"] = data_entry.get('api_data', {}).get('api_name', 'No api_name found')
                output_dict["description"] = data_entry.get('api_data', {}).get('description', 'No description found')
# print(safe_description)

                output_data.append(output_dict)
        
    output_path = f"{output_path}/{extraction_domain}_tool.json"
    
    with open(output_path, "w", encoding='utf-8') as file:
        json.dump(output_data, file, indent=4)

# get the keyword_dataset
def get_keyword_dataset(dataset_path: str):
    
    """
    return: 
    """
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    return dataset
        
# get the index dataset which includes best_demo and target_demo.
def get_target_demo_dataset(dataset_path: str):
    """
    Args:

    Return:

    """
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    return dataset

def process_tool_selection_dataset():
    """
    
    """

def process_SEO_dataset():
    """
    
    """

def get_integrate_keywords_dataset(keywords_dataset_path1, keywords_dataset_path2):
    
    # keywords1 = load_dataset("json", data_files=keywords_dataset_path1, split="train")
    # keywords2 = load_dataset("json", data_files=keywords_dataset_path2, split="train")

    # merged_dataset = concatenate_datasets([keywords1, keywords2])
    output_json_path = "/home/lzs/Comattack/src/data/new_keywords_Qwen3.json"
    # merged_dataset.to_json(output_json_path, force_ascii=False, indent=4)

    with open(keywords_dataset_path1, 'r', encoding='utf-8') as f1:
        list1 = json.load(f1)
    
    with open(keywords_dataset_path2, 'r', encoding='utf-8') as f2:
        list2 = json.load(f2)
    
    if isinstance(list1, list) and isinstance(list2, list):
        merged_list = list1 + list2

    with open(output_json_path, 'w', encoding='utf-8') as f_out:
        json.dump(merged_list, f_out, ensure_ascii=False, indent=4)

if __name__ == "__main__":

    get_integrate_keywords_dataset(
        keywords_dataset_path1="/home/lzs/Comattack/src/data/revised_keywords_with_Qwen3_1.json",
        keywords_dataset_path2="/home/lzs/Comattack/src/data/revised_keywords_with_Qwen3_2.json"
    )