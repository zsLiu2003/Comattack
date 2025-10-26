import torch
import transformers
from datasets import load_dataset, Dataset, concatenate_datasets
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm

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

def process_tool_selection_dataset(dataset_path):
    """
    """

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    return dataset

def process_SEO_dataset():
    """
    
    """

def get_integrate_keywords_dataset(keywords_dataset_path1, keywords_dataset_path2):
    
    # keywords1 = load_dataset("json", data_files=keywords_dataset_path1, split="train")
    # keywords2 = load_dataset("json", data_files=keywords_dataset_path2, split="train")

    # merged_dataset = concatenate_datasets([keywords1, keywords2])
    output_json_path = "src/data/new_keywords_Qwen3.json"
    # merged_dataset.to_json(output_json_path, force_ascii=False, indent=4)

    with open(keywords_dataset_path1, 'r', encoding='utf-8') as f1:
        list1 = json.load(f1)
    
    with open(keywords_dataset_path2, 'r', encoding='utf-8') as f2:
        list2 = json.load(f2)
    
    if isinstance(list1, list) and isinstance(list2, list):
        merged_list = list1 + list2

    with open(output_json_path, 'w', encoding='utf-8') as f_out:
        json.dump(merged_list, f_out, ensure_ascii=False, indent=4)

def get_QA_dataset(dataset_path: str):

    df = pd.read_parquet(dataset_path)
    print("-----------Successfully read the dataset!------------")
    
    print("------------The first five data entries in the dataset--------------")
    print(df.head())

    print("-"*10 + "The detailed information of this dataset." + "-"*10)
    df.info()

    print("-"*10 + "The number of column and row." + "-"*10)
    print(df.shape)

    """Get the subset of squad: context, question, and answers."""
    subset = df[["context", "question", "answers"]]
    
    # data_dict_list = subset.to_dict(orient="records")
    
    # dataset = load_dataset("parquet", data_files=dataset_path, split="train")
    
    dataset = df.to_dict(orient="records")
    
    return dataset
    # for data in dataset[:5]:
    #     for key, value in data.items():
    #         print(f"key={key}, value={value}")
    # # output_path = f"{output_path}/squad_QA_dataset.json"
    # with open(output_path, "w", encoding="utf-8") as file:
    #     json.dump(dataset, file, indent=4)
    # print(data_dict[1]["answers"])

        # print(data)
                
    # return df

# def get_recommendation_test_dataset(dataset_path):
    
#     dataset = load_dataset("json", data_files=dataset_path, split="train")
    
#     print("----------Process the recommendation dataset.----------")
#  
#    for data_entry in tqdm(dataset):


def get_confused_recommendation_dataset(dataset_path: str):
    print(f"----------Process the {dataset_path}.----------")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    # model = GPT2LMHeadModel.from_pretrained(self.compression_model_name, device_map='auto')
    # tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    # model.eval()

    output_list = []
    le = 20
    result = 0
    dict_num = 0
    real_num = 0
    for data_entry in tqdm(dataset):
        output_dict = {}
        for key, value in data_entry.items():
            output_dict[key] = {
                "original": value["original"],
                "replaced": value["new"],
            }
        
        output_list.append(output_dict)
    
    output_path = "src/data/replaced_confused_recommendation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_list, f, indent=4, ensure_ascii=False)


def process_keyword_dataset(dataset_path: str, output_path: str):

    def replace_infinity(obj):
        if isinstance(obj, dict):
            return {k: replace_infinity(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_infinity(item) for item in obj]
        elif isinstance(obj, str) and obj.strip() == "Infinity":
            return 0
        else:
            return obj
        

    """
    Process the keyword dataset to get the keywords and their corresponding demo.
    """
    # dataset = load_dataset("json", data_files=dataset_path, split="train")  

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = replace_infinity(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)


def fill_missing_keywords(dataset1_path: str, dataset2_path: str, output_path: str):
    """
    Fill the missing keywords in the dataset.
    """
    filled = []
    with open(dataset1_path, "r", encoding="utf-8") as f1, open(dataset2_path, "r", encoding="utf-8") as f2:
        dataset1 = json.load(f1)
        dataset2 = json.load(f2)
    for i, (d1, d2) in enumerate(zip(dataset1, dataset2)):
        merged_entry = d2.copy()
        for key in d1:
            if "output" in key:
                index = str(key[6])
                new_key = f"demo{index}"
                if new_key not in merged_entry:
                    merged_entry[new_key] = {
                        "original": d1[key],
                        "replaced": d1[key],
                        "original_keyword": "",
                        "replaced_keyword": "",
                        "original_keyword_ppl": 0,
                        "replaced_keyword_ppl": 0, # Assuming you want to keep the original as replaced
                    }
        filled.append(merged_entry)
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(filled, f_out, indent=4, ensure_ascii=False)
        
    print(f"Filled dataset saved to {output_path}")

def QA_dataset_process(dataset1_path, dataset2_path):
    
    dataset1 = load_dataset("json", data_files=dataset1_path, split="train")
    dataset2 = load_dataset("json", data_files=dataset2_path, split="train")

    output_list = []
    for data_entry1, data_entry2 in zip(dataset1, dataset2):
        output_dict = {}
        output_dict["original"] = data_entry1["original"]
        output_dict["replaced"] = data_entry2["optimized"]

        output_list.append(output_dict)

    output_path = "src/data/QA_Stealth.json"
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output_list, file, indent=4)

if __name__ == "__main__":

    # get_integrate_keywords_dataset(
    #     keywords_dataset_path1="src/data/revised_keywords_with_Qwen3_1.json",
    #     keywords_dataset_path2="src/data/revised_keywords_with_Qwen3_2.json"
    # )
    # output_path = "src/data/new_keywords_decrease_3.json"
    # dataset1_path = "src/data/data.json"
    # dataset_path = "src/data/new_keywords_decrease_2.json"
    # # process_keyword_dataset(dataset_path=dataset_path, output_path=output_path)
    # fill_missing_keywords(
    #     dataset1_path=dataset1_path,
    #     dataset2_path=dataset_path,
    #     output_path=output_path
    # )
    # get_keyword_dataset(dataset_path="src/data/data.json"
    # get_confused_recommendation_dataset(dataset_path="src/data/confused_recommendation.json")
    
    dataset1_path = "src/data/QA_original_results2.json"
    dataset2_path = "src/data/QA_optimized_results.json"

    QA_dataset_process(
        dataset1_path=dataset1_path,
        dataset2_path=dataset2_path,
    )