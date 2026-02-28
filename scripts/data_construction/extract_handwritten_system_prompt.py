from datasets import load_dataset
import json
import re

def filte_tool_description(prompt: str):
    if not prompt:
        return prompt

    p = prompt

    # 兼容：有些数据里是字面量 "\n" 而不是实际换行
    if "\\n" in p and "\n" not in p:
        p = p.replace("\\n", "\n")

    p = p.replace("\r\n", "\n")

    # 优先策略：从 **Procedure** 后面找第一段 "You are ..."
    proc_idx = p.find("**Procedure**")
    if proc_idx != -1:
        tail = p[proc_idx:]
        m = re.search(r"\n\s*\n(You are\b[\s\S]*)", tail)  # 找到 Procedure 后第一个以 You are 开头的段落
        if m:
            return m.group(1).strip()

    # 兜底策略：保留“最后一次出现的段落级 You are ...”（通常就是最终 persona）
    matches = list(re.finditer(r"(?:^|\n\s*\n)(You are\b[\s\S]*)", p))
    if matches:
        return matches[-1].group(1).strip()

    # 再兜底：原样返回
    return p.strip()
    

def extract_system_prompt_from_SFT_subset(dataset_name: str):
    ds = load_dataset(dataset_name, "train_sft", split="train")
    seen = set()
    def keep_first(example):
        
        i = example["system_id"]
        if i in seen:
            return False
        seen.add(i)
        return True
    
    new_ds = ds.filter(keep_first)

    new_dataset = []
    for id,item in enumerate(new_ds):
        system_prompt = filte_tool_description(item["messages"][0]["content"])
        adversarial_query_list = []
        id = id - 1
        for i in range(id*10+5, id*10+10):
            message_list = ds[i]["messages"]
            adversarial_query_list.append(message_list[1]["content"])

        new_item = {
            "system_prompt": system_prompt,
            "adversarial_query_list": adversarial_query_list
        }
        new_dataset.append(new_item)
    
    return new_dataset

def extract_dataset(dataset_name: str):
    ds = load_dataset(dataset_name, "handwritten", split="train")
    seen = set()
    def keep_first(example):
        
        i = example["id"]
        if i in seen:
            return False
        seen.add(i)
        return True
    
    new_ds = ds.filter(keep_first)

    new_dataset = []
    for id, item in enumerate(new_ds):
        system_prompt_id = item["id"]
        system_prompt = item["messages"][0]["content"]
        ds_same = ds.filter(lambda x: x["id"] == system_prompt_id)
        all_messages = ds_same["messages"]
        adversarial_query_list = []
        for messages in all_messages:
            adversarial_query_list.append(messages[1]["content"])
        
        new_item = {
            "system_prompt": system_prompt,
            "adversarial_query_list": adversarial_query_list
        }
        new_dataset.append(new_item)
        # guardrail_list = []
        # for guardrail in item["guardrails"]:
        #     guardrail_list.append({
        #         "keyword": "",
        #         "sentence": guardrail
        #     })
        
        # system_prompt = message_list[0]["content"]
        
        # new_item = {
        #     "system_prompt": system_prompt,
        #     "guardrail_list": guardrail_list,
        #     # "adversarial_query": adversarial_query
        # }
        # new_dataset.append(new_item)
    
    return new_dataset

if __name__ == "__main__":
    # dataset = extract_dataset("normster/SystemCheck")

    # with open("data/handwritten_system_prompt_adversarial_queries.json", "w", encoding="utf-8") as f:
    #     json.dump(dataset, f, indent=2, ensure_ascii=False,)

    dataset = extract_system_prompt_from_SFT_subset("normster/SystemCheck")

    with open("data/sft_system_prompt_adversarial_queries.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False,)