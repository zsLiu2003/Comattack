def extract_attacked_keywords(modified_sentence: str, K_true: set) -> set:

    modified_sentence_lower = modified_sentence.lower()
    return {kw for kw in K_true if kw.lower() in modified_sentence_lower}


def compute_keyword_f1(K_true: set, K_attack: set):

    true_positive = len(K_true & K_attack) 

    precision = true_positive / len(K_attack) if len(K_attack) > 0 else 0.0
    recall = true_positive / len(K_true) if len(K_true) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


from sklearn.metrics import precision_recall_fscore_support

from datasets import load_dataset
if __name__ == "__main__":

    keyword_dataset_path = "src/data/new_keywords_Qwen3.json"
    keyword_dataset = load_dataset("json", data_files=keyword_dataset_path, split="train")

    dataset_path_list = [
        "src/data/replaced_confused_recommendation.json",
        "src/data/replaced_ppl_adjective_increase.json",
        "src/data/replaced_ppl_connectors_decrease.json",
        "src/data/replaced_ppl_prep_context_decrease.json",
        "src/data/replaced_ppl_synonym_decrease.json",
        "src/data/replaced_ppl_synonym_increase.json",
    ]  

    for dataset_path in dataset_path_list:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        F1_score = 0
        count = 0
        for keywords, demo_data in zip(keyword_dataset, dataset):
            for (key1, value1), (key2, value2) in zip(keywords.items(), demo_data.items()):
                key1 = key2
                keyword_set = set(value1)
                keyword_attack = extract_attacked_keywords(
                    modified_sentence=value2["replaced"],
                    K_true=keyword_set,
                )
                _,_,temp = compute_keyword_f1(
                    K_true=keyword_set,
                    K_attack=keyword_attack,
                )
                F1_score += temp
                count += 1
        F1_score /= count
        print(f"The F1-Score of {dataset_path} is: {F1_score}")
        
    