from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.get_prompt import get_defense_prompt
from datasets import load_dataset

def llm_inference(model=None, tokenizer=None, text=str):
    
    messages = tokenizer.apply_chat_template(
        text,
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
    
    print(content)
    if "True" in content:
        return True
    elif "False" in content:
        return False
    else:
        return None
    
if __name__ == "__main":

    dataset_path_list = [
        "/home/lzs/Comattack/src/data/replaced_confused_recommendation.json",
        "/home/lzs/Comattack/src/data/replaced_ppl_adjective_increase.json",
        "/home/lzs/Comattack/src/data/replaced_ppl_connectors_decrease.json",
        "/home/lzs/Comattack/src/data/replaced_ppl_prep_context_decrease.json",
        "/home/lzs/Comattack/src/data/replaced_ppl_synonym_decrease.json",
        "/home/lzs/Comattack/src/data/replaced_ppl_synonym_increase.json",
    ]
    model_name = "/opt/model/Qwen3-32B"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    QA_dataset_path = "/home/lzs/Comattack/src/data/QA_keywords_edit.json"


    for dataset_path in dataset_path_list:
        count = 0
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        
        for data_entry in dataset:
            for key, value in data_entry.items():
                text = value["replaced"]
                ans = llm_inference(
                    model=model,
                    tokenizer=tokenizer,
                    text=text,
                )
                if ans is not None and ans == False:
                    count += 1
        count /= 500
        print(f"Detection rate of {dataset_path} is: {count}")
    
                    
    