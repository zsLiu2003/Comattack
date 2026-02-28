from datasets import load_dataset
import re

def extract_number(value):
    # 如果是字典，提取 content
    if isinstance(value, dict):
        value = value.get("content", "")
    # 提取字符串中的数字
    match = re.search(r'\d+', str(value))
    return match.group() if match else None  # 返回数字字符串，如 '5'


def get_results(dataset_path):
    
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    count = 0
    for d in dataset:
        
        contents = [extract_number(v) for v in d.values()]
        if len(set(contents)) > 1:
            count += 1

    
    print(f"results of {dataset_path}: {count}")
        

if __name__ == "__main__":
    dataset_path_list = [
        "src/data/confused_compressed_llama3.json",
        "src/data/confused_compressed_phi4.json",
        "src/data/confused_compressed_qwen3.json",
        # "src/data/confused_recommendation.json",
        "src/data/confused_without_compressed_llama3.json",
        "src/data/confused_without_compressed_qwen3.json",
        "src/data/decrease_compressed_llama3.json",
        "src/data/decrease_compressed_phi4.json",
        "src/data/decrease_compressed_qwen3.json",
        # "src/data/decrease_recommendation.json",
        "src/data/decrease_without_compressed_llama3.json",
        "src/data/decrease_without_compressed_qwen3.json",
        "src/data/increase_compressed_llama3.json",
        "src/data/increase_compressed_phi4.json",
        "src/data/increase_compressed_qwen3.json",
        # "src/data/increase_recommendation.json",
        "src/data/increase_without_compressed_llama3.json",
        "src/data/increase_without_compressed_qwen3.json",
        "src/data/keyword_compressed_llama3.json",
        "src/data/keyword_compressed_phi4.json",
        "src/data/keyword_compressed_qwen3.json",
        # "src/data/keyword_recommendation.json",
        "src/data/keyword_without_compressed_llama3.json",
        "src/data/keyword_without_compressed_qwen3.json",
        "src/data2/increase_compressed_llama3.json",
        "src/data2/increase_compressed_phi4.json",
        "src/data2/increase_compressed_qwen3.json",
        # "src/data2/increase_recommendation.json",
        "src/data2/increase_without_compressed_llama3.json",
        "src/data2/increase_without_compressed_qwen3.json",
    ]

    for dataset_path in dataset_path_list:
        get_results(dataset_path)
    
    print("All results have been processed.")
    print("Please check the results in the terminal.")