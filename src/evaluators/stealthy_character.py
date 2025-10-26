import torch
import Levenshtein  # pip install python-Levenshtein
from sentence_transformers import SentenceTransformer, util

sentence_model = SentenceTransformer('models/all-mpnet-base-v2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_normalized_edit_similarity(s_adv: str, s_orig: str) -> float:
    edit_dist = Levenshtein.distance(s_adv, s_orig)
    max_len = max(len(s_adv), len(s_orig))
    return 1 - (edit_dist / max_len) if max_len > 0 else 1.0

def compute_semantic_cosine_similarity(s_adv: str, s_orig: str) -> float:
    """
    """
    emb_adv = sentence_model.encode(s_adv, convert_to_tensor=True, device=device)
    emb_orig = sentence_model.encode(s_orig, convert_to_tensor=True, device=device)
    return util.pytorch_cos_sim(emb_adv, emb_orig).item()

def compute_stealth_score(s_adv: str, s_orig: str, lambda_weight: float = 0.5) -> tuple:
    """
    """
    char_sim = compute_normalized_edit_similarity(s_adv, s_orig)
    semantic_sim = compute_semantic_cosine_similarity(s_adv, s_orig)
    stealth_score = lambda_weight * char_sim + (1 - lambda_weight) * semantic_sim
    return stealth_score, char_sim, semantic_sim


from datasets import load_dataset
from tqdm import tqdm

if __name__ == "__main__":
    
    dataset_path_list = [
        # "src/data/QA_Stealth.json",
        "src/data/replaced_confused_recommendation.json",
        "src/data/replaced_ppl_adjective_increase.json",
        "src/data/replaced_ppl_connectors_decrease.json",
        "src/data/replaced_ppl_prep_context_decrease.json",
        "src/data/replaced_ppl_synonym_decrease.json",
        "src/data/replaced_ppl_synonym_increase.json",
    ]
    
    for dataset_path in dataset_path_list:
        dataset = load_dataset("json", data_files=dataset_path, split="train")

        stealth, first, second = 0,0,0
        count = 0
        for data_entry in tqdm(dataset):
            for key, value in data_entry.items():
                temp1, temp2, temp3 = compute_stealth_score(
                    s_adv=value["replaced"],
                    s_orig=value["original"],
                )
                stealth += temp1
                first += temp2
                second += temp3
                count += 1

        stealth /= count
        first /= count
        second /= count
            
        print(f"The stealth, first, second of {dataset_path} are {stealth}, {first}, {second}")