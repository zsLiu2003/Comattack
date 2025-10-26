from llmlingua import PromptCompressor
from datasets import load_dataset
# from src.utils.
dataset = load_dataset("json", data_files="src/data/data.json", split="train")

print(dataset)

model_name = "models/gpt2-dolly"
device="cuda:0"
compression_model = PromptCompressor(
        model_name=model_name,
        device_map=device
    )

data = dataset[0]
prompt = str(data["output1"] + data["output2"])
prompt_list = [data["output1"], data["output2"], data["output3"], data["output4"], data["output5"]]
compressed_data = compression_model.compress_prompt(
    prompt_list,
    instruction="",
    question="",
    target_token=50,
)
print(len(prompt_list), len(compressed_data["compressed_prompt_list"]))
print(f"origin={prompt_list} \n--- compressed={compressed_data}")