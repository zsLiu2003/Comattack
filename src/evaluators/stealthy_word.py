from sentence_transformers import SentenceTransformer, util
from bert_score import BERTScorer
import torch

def stealth_score(original: str, adversarial: str, lambda_weight: float = 0.5) -> float:
    """
    Compute Stealth(C, C~) = λ · cosine(sim(C, C~)) + (1 - λ) · BERTScore(C, C~)

    Args:
        original: Original sentence C
        adversarial: Modified sentence C~
        lambda_weight: Weight between [0, 1] controlling trade-off

    Returns:
        Composite similarity score (higher = more stealthy)
    """
    # ===== Sentence-level similarity =====
    sentence_model = SentenceTransformer('models/all-mpnet-base-v2')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emb1 = sentence_model.encode(original, convert_to_tensor=True, device=device)
    emb2 = sentence_model.encode(adversarial, convert_to_tensor=True, device=device)
    cosine_sim = util.pytorch_cos_sim(emb1, emb2).item()

    # ===== BERTScore with local model =====
    scorer = BERTScorer(
        model_type='bert-base-uncased',  # ✅ 本地路径
        lang='en',
        rescale_with_baseline=True,
        device=device
    )
    P, R, F1 = scorer.score([adversarial], [original])
    bert_score_f1 = F1.item()

    # ===== Composite stealth score =====
    stealth = lambda_weight * cosine_sim + (1 - lambda_weight) * bert_score_f1
    return stealth, cosine_sim, bert_score_f1

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
                temp1, temp2, temp3 = stealth_score(
                    adversarial=value["replaced"],
                    original=value["original"],
                )
                stealth += temp1
                first += temp2
                second += temp3
                count += 1

        stealth /= count
        first /= count
        second /= count
            
        print(f"The stealth, first, second of {dataset_path} are {stealth}, {first}, {second}")