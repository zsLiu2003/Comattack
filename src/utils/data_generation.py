from src.data.data_process import get_tool_selection_dataset
extraction_data = ["Feature Extraction", "Text-to-Image"]

dataset_path = "src/data/huggingface_train.jsonl"
output_path = "src/data"

for extraction_domain in extraction_data:
    get_tool_selection_dataset(
        extraction_domain=extraction_domain,
        dataset_path=dataset_path,
        output_path=output_path
    )
