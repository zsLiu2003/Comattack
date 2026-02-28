import json
from datasets import load_dataset
from tqdm import tqdm

# dataset = load_dataset("json", data_files="/home/zliuhi/SystemCom/data/consolidated_guardrails/consolidated_guardrails.json", split="train")
with open("/home/zliuhi/SystemCom/data/extracted_guardrails_reduced.json", "r") as f:
    dataset = json.load(f)
# data = dataset["prompts"]
new_dataset = []
guardrail_all = 0
guardrail_matched = 0
for item in tqdm(dataset):
    new_item = {}
    system_prompt = item["reduced_system_prompt"]
    new_item["system_prompt"] = system_prompt
    guardrails = item["guardrail_list"]
    guardrail_list = []
    for guardrail in guardrails:
        guardrail_all += 1
        if guardrail["sentence"] in system_prompt:
            guardrail_matched += 1
            # new_item["guardrail_matched"] = guardrail["sentence"]
            guardrail_list.append(
                {
                    "keyword": guardrail["keyword"],
                    "sentence": guardrail["sentence"]
                }
            )
    new_item["guardrail_list"] = guardrail_list
    new_dataset.append(new_item)

with open("/home/zliuhi/SystemCom/data/system_prompt_guardrails_reduced_filtered.json", "w") as f:
    json.dump(new_dataset, f, indent=4)
print(f"guardrail_all: {guardrail_all}")
print(f"guardrail_matched: {guardrail_matched}")
print(f"guardrail_matched / guardrail_all: {guardrail_matched / guardrail_all}")
