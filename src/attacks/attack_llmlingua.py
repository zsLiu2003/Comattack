from llmlingua import  PromptCompressor
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from datasets import load_dataset
class LLMLinguaAttack():
    
    def __init__(self, prompt_dataset_name: str) -> None:
        self.model_name = "lgaalves/gpt2-dolly"
        self.llmlingua = PromptCompressor(self.model_name, device_map="cuda:7")
        self.data = load_dataset(prompt_dataset_name, split="attack")
        
    # def attack(self):
