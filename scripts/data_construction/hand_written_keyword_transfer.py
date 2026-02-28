from datasets import load_dataset

import json

dataset = json.load(open("/home/zliuhi/SystemCom/data/guardrail_violation_queries_for_handwritten_system_prompt.json", "r", encoding="utf-8"))
new_dataset = []
for item_dict in dataset:
    guardrail_list = item_dict["guardrail_list"]
    new_guardrail_list = []
    for guardrail_dict in guardrail_list:
        new_guardrail_dict = {
            "keyword": guardrail_dict["adversarial_generation"]["keyword"],
            "sentence": guardrail_dict["sentence"],
            "adversarial_query": guardrail_dict["adversarial_query"],
            "adversarial_generation": {
                "adversarial_query": guardrail_dict["adversarial_generation"]["adversarial_query"],
                "target_violation": guardrail_dict["adversarial_generation"]["target_violation"],
            }
        }
        new_guardrail_list.append(new_guardrail_dict)
    
    new_data_dict = {
        "system_prompt": item_dict["system_prompt"],
        "guardrail_list": new_guardrail_list,
    }
    new_dataset.append(new_data_dict)
with open("/home/zliuhi/SystemCom/data/guardrail_violation_queries_for_handwritten_system_prompt_v2.json", "w", encoding="utf-8") as f:
    json.dump(new_dataset, f, ensure_ascii=False, indent=4)

