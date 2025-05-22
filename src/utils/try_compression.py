from llmlingua import PromptCompressor
from datasets import load_dataset
# from src.utils.
dataset = load_dataset("json", data_files="/home/lzs/Comattack/src/data/data.json", split="train")

print(dataset)

model_name = "/opt/model/models/gpt2-dolly"
device="cuda:0"
compression_model = PromptCompressor(
        model_name=model_name,
        device_map=device
    )

data = dataset[0]
prompt = str(data["output1"] + data["output2"])
compressed_data = compression_model.compress_prompt(
    prompt,
    instruction="",
    question="",
    target_token=50,
)
print(len(prompt), len(compressed_data["compressed_prompt"]))
print(f"origin={prompt} \n--- compressed={compressed_data}")