import torch
import transformers
from datasets import load_dataset, Dataset
import json
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
                # if int(k) == 6:
                #     break
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
                new_data[f"demo_{k}"] = value
            else:
                new_data[key] = value
            
        new_dataset.append(new_data)
    
    return Dataset.from_list(new_dataset)

def get_tool_selection_dataset(extraction_domain, dataset_path, output_path):

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


