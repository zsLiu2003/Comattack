from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.get_prompt import get_defense_prompt
from datasets import load_dataset
from tqdm import tqdm


def llm_inference(model=None, tokenizer=None, text=str):
    
    prompt = get_defense_prompt()
    messages = [
        {"role":"system", "content": prompt},
        {"role": "user", "content": text} 
    ]
    messages = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer(
        messages,
        return_tensors='pt',
        padding=True,
    ).to(model.device)

    generate_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768,
    )
    
    output_ids = generate_ids[0][len(model_inputs.input_ids[0]):].tolist()
    del generate_ids
    
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    
    # print(content)
    if "True" in content:
        return True
    elif "False" in content:
        return False
    else:
        return None
    
if __name__ == "__main__":

    dataset_path_list = [
        # "src/data/replaced_confused_recommendation.json",
        # "src/data/replaced_ppl_adjective_increase.json",
        # "src/data/replaced_ppl_connectors_decrease.json",
        # "src/data/replaced_ppl_prep_context_decrease.json",
        "src/data/replaced_ppl_synonym_decrease.json",
        "src/data/replaced_ppl_synonym_increase.json",
    ]
    model_name = "models/Qwen3-32B"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    QA_dataset_path = "src/data/QA_keywords_edit.json"


    # count = 0
    # dataset = load_dataset("json", data_files=QA_dataset_path, split="train")
    
    # for data_entry in tqdm(dataset):
    #     text = data_entry["optimized"]
    #     ans = llm_inference(
    #         model=model,
    #         tokenizer=tokenizer,
    #         text=text,
    #     )
    #     if ans is not None and ans == False:
    #         count += 1
    # count /= 283
    # print(f"Detection rate of {QA_dataset_path} is: {count}")
    for dataset_path in dataset_path_list:
        count = 0
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        
        for data_entry in tqdm(dataset):
            for key, value in data_entry.items():
                text = value["original"]
                ans = llm_inference(
                    model=model,
                    tokenizer=tokenizer,
                    text=text,
                )
                if ans is not None and ans == False:
                    count += 1
        count /= 500
        print(f"Detection rate of {dataset_path} is: {count}")
    
                    
    