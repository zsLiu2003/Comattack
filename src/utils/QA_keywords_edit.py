from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.get_edit_token import EditPrompt
from src.data.data_process import get_QA_dataset
from tqdm import tqdm
import json
import torch
from datasets import load_dataset


def edit_QA_keywords(
    model_name,
    dataset_path,
    output_path,
):
    dataset = get_QA_dataset(
        dataset_path=dataset_path
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    Edit = EditPrompt(
        dataset=dataset,
        model_name=model_name,
        phrase_model_name=""
    )

    QA_list = []
    for idx, data in tqdm(enumerate(dataset), desc="---------Process the QA dataset by editing keywords.----------"):
        sentence = data["context"]
        QA_dict = {}

        optimized_sentence = sentence
        optimized_keyword = ""
        optimized_ppl = 0
        keyword_list = []
        for keyword in data["answers"]["text"]:
            words = keyword.split()
            if len(words) < 3:
                for word in words:
                    keyword_list.append(word)
        unique_keywords = set(keyword_list)
        unique_list = list(unique_keywords)
        for id, keyword in enumerate(unique_list):
            
            optimized_sentence, optimized_keyword, optimized_ppl = Edit.optimize_with_character_edits(
                model=model,
                tokenizer=tokenizer,
                sentence=sentence,
                target_word=keyword,
            )
            
            # if id+1 == len(data["answer"]["text"]):
        
        if optimized_sentence != sentence:
            QA_dict["idx"] = idx
            QA_dict["original"] = sentence
            QA_dict["optimized"] = optimized_sentence

            QA_list.append(QA_dict)
        
    output_path = f"{output_path}/QA_keywords_edit.json"
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(QA_list, file, indent=4)
    
    print("--------------------Successfully finised editing the keywords in QA dataset!--------------------")

if __name__ == "__main__":

    model_name = "models/gpt2-dolly"
    dataset_path = "squad/validation-00000-of-00001.parquet"
    output_path = "src/data"

    edit_QA_keywords(
        model_name=model_name,
        dataset_path=dataset_path,
        output_path=output_path,
    )
        
            
                 
        