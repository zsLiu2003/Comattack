def extract_attacked_keywords(modified_sentence: str, K_true: set) -> set:
    """
    在修改后的句子中寻找仍然保留的关键词。

    参数：
        modified_sentence: 改动后的句子
        K_true: 原始关键词集合

    返回：
        K_attack: 修改后句子中保留下来的关键词集合
    """
    modified_sentence_lower = modified_sentence.lower()
    return {kw for kw in K_true if kw.lower() in modified_sentence_lower}


def compute_keyword_f1(K_true: set, K_attack: set):
    """
    计算关键词的 Precision, Recall 和 F1 分数。

    参数:
        K_true: 原始上下文中的关键词集合
        K_attack: 攻击后保留下来的关键词集合

    返回:
        precision, recall, f1
    """
    true_positive = len(K_true & K_attack)  # 交集的大小

    precision = true_positive / len(K_attack) if len(K_attack) > 0 else 0.0
    recall = true_positive / len(K_true) if len(K_true) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


from sklearn.metrics import precision_recall_fscore_support

# def word_level_f1(original_sentence: str, attacked_sentence: str):
#     """
#     计算基于word-level差异的F1分数，衡量修改词的检测效果。
#     """
#     original_words = original_sentence.strip().split()
#     attacked_words = attacked_sentence.strip().split()
    
#     # 对齐：取最小长度
#     min_len = min(len(original_words), len(attacked_words))
#     original_words = original_words[:min_len]
#     attacked_words = attacked_words[:min_len]

#     # 真实标签（1 表示被修改）
#     true_modified = [int(o != a) for o, a in zip(original_words, attacked_words)]

#     # 假设攻击者也知道哪些词修改了（这里我们假设实际攻击行为本身 = 预测），实际使用时可以替换为模型预测
#     predicted_modified = true_modified  # 如果有预测，可以换成 model 的输出

#     # 计算 Precision / Recall / F1
#     precision, recall, f1, _ = precision_recall_fscore_support(
#         true_modified, predicted_modified, average='binary', zero_division=0
#     )

#     return precision, recall, f1, sum(true_modified)


from datasets import load_dataset
if __name__ == "__main__":

    keyword_dataset_path = "/home/lzs/Comattack/src/data/new_keywords_Qwen3.json"
    keyword_dataset = load_dataset("json", data_files=keyword_dataset_path, split="train")

    dataset_path_list = [
        "/home/lzs/Comattack/src/data/replaced_confused_recommendation.json",
        "/home/lzs/Comattack/src/data/replaced_ppl_adjective_increase.json",
        "/home/lzs/Comattack/src/data/replaced_ppl_connectors_decrease.json",
        "/home/lzs/Comattack/src/data/replaced_ppl_prep_context_decrease.json",
        "/home/lzs/Comattack/src/data/replaced_ppl_synonym_decrease.json",
        "/home/lzs/Comattack/src/data/replaced_ppl_synonym_increase.json",
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
        
    