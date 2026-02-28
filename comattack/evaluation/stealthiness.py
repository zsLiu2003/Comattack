"""
Stealthiness metrics for adversarial text perturbations.

Two scoring modes:
  - Word-level : cosine similarity (SentenceTransformer) + BERTScore
  - Char-level : normalized edit distance + cosine similarity
"""
import torch


# ── Word-level stealth (cosine + BERTScore) ───────────────────────────────

def stealth_score(
    original: str,
    adversarial: str,
    lambda_weight: float = 0.5,
    sentence_model_name: str = "all-mpnet-base-v2",
    bert_model_type: str = "bert-base-uncased",
) -> tuple:
    """
    Compute Stealth(C, C~) = lambda * cosine_sim + (1 - lambda) * BERTScore_F1.

    Args:
        original: Original sentence C
        adversarial: Modified sentence C~
        lambda_weight: Weight between [0, 1] controlling trade-off
        sentence_model_name: SentenceTransformer model name or path
        bert_model_type: BERTScorer model type

    Returns:
        (stealth_score, cosine_sim, bert_score_f1)
    """
    from sentence_transformers import SentenceTransformer, util
    from bert_score import BERTScorer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sentence_model = SentenceTransformer(sentence_model_name)
    emb1 = sentence_model.encode(original, convert_to_tensor=True, device=device)
    emb2 = sentence_model.encode(adversarial, convert_to_tensor=True, device=device)
    cosine_sim = util.pytorch_cos_sim(emb1, emb2).item()

    scorer = BERTScorer(
        model_type=bert_model_type,
        lang="en",
        rescale_with_baseline=True,
        device=device,
    )
    P, R, F1 = scorer.score([adversarial], [original])
    bert_score_f1 = F1.item()

    stealth = lambda_weight * cosine_sim + (1 - lambda_weight) * bert_score_f1
    return stealth, cosine_sim, bert_score_f1


# ── Char-level stealth (edit distance + cosine) ──────────────────────────

def compute_normalized_edit_similarity(s_adv: str, s_orig: str) -> float:
    """Normalized edit-distance similarity in [0, 1]."""
    import Levenshtein

    edit_dist = Levenshtein.distance(s_adv, s_orig)
    max_len = max(len(s_adv), len(s_orig))
    return 1 - (edit_dist / max_len) if max_len > 0 else 1.0


def compute_semantic_cosine_similarity(
    s_adv: str,
    s_orig: str,
    sentence_model_name: str = "all-mpnet-base-v2",
) -> float:
    """Cosine similarity between sentence embeddings."""
    from sentence_transformers import SentenceTransformer, util

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentence_model = SentenceTransformer(sentence_model_name)
    emb_adv = sentence_model.encode(s_adv, convert_to_tensor=True, device=device)
    emb_orig = sentence_model.encode(s_orig, convert_to_tensor=True, device=device)
    return util.pytorch_cos_sim(emb_adv, emb_orig).item()


def compute_stealth_score(
    s_adv: str,
    s_orig: str,
    lambda_weight: float = 0.5,
    sentence_model_name: str = "all-mpnet-base-v2",
) -> tuple:
    """
    Char-level stealth: lambda * edit_sim + (1 - lambda) * cosine_sim.

    Returns:
        (stealth_score, char_similarity, semantic_similarity)
    """
    char_sim = compute_normalized_edit_similarity(s_adv, s_orig)
    semantic_sim = compute_semantic_cosine_similarity(
        s_adv, s_orig, sentence_model_name=sentence_model_name
    )
    score = lambda_weight * char_sim + (1 - lambda_weight) * semantic_sim
    return score, char_sim, semantic_sim
