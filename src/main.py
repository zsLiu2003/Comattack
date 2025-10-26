from src.utils.get_edit_token import EditPrompt
from src.data.data_process import get_common_compression_dataset, get_keyword_dataset, get_target_demo_dataset,get_pure_demo_dataset
from datasets import load_dataset

if __name__ == "__main__":
    

    # name of the ppl model and phrase model
    ppl_model_name = "models/gpt2-dolly"
    phrase_model_name = "models/Qwen3-32B"


    # get the demo dataset
    dataset_path = "src/data/data.json"
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = get_pure_demo_dataset(dataset=dataset)

    # get the keywords dataset
    keywords_dataset_path = "src/data/new_keywords_Qwen3.json"
    # keywords_dataset = get_keyword_dataset(
    #     dataset_path=keywords_dataset_path,
    # )
    # keywords_dataset = load_dataset("json", data_files=keywords_dataset_path, split="train")

    # get target and best demo dataset 
    target_demo_path = "src/data/data_best_Qwen3.json"
    # target_demo_dataset = get_target_demo_dataset(
    #     dataset_path=target_demo_path,
    # )

    # output path
    output_path = "src/data"

    # edit the demo
    Edit = EditPrompt(
        dataset=dataset,
        model_name=ppl_model_name,
        phrase_model_name=phrase_model_name,
    )
    
    Edit.get_edit_tokens(
        keywords_dataset_path=keywords_dataset_path,
        target_demo_path=target_demo_path,
        output_path=output_path,
        top_k=10, 
    )