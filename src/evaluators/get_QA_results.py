from datasets import load_dataset

def get_QA_results(dataset_path):  

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    count = 0
    for data_entry in dataset:
        if data_entry["real_answer"] == data_entry["answer"] or data_entry["real_answer"] in data_entry["answer"]:
            count += 1
    
    print(f"Results of {dataset_path}: {count} out of {len(dataset)}")
    print(f"Accuracy: {count / len(dataset)}")
    return count, count / len(dataset)

if __name__ == "__main__":
    dataset_path = "src/data/QA_optimized_results2.json"
    get_QA_results(dataset_path)