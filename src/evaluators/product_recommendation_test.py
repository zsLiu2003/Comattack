from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
from datasets import load_dataset
from llmlingua import PromptCompressor

class Product_recommendation():

    def __init__(self, compression_model_name, large_model_name, dataset_path, question_dataset_path, device="cuda:0"):
        self.compression_model_name = compression_model_name
        self.large_model_name = large_model_name
        self.compression_model = PromptCompressor(
            model_name=compression_model_name,
            device_map=device,
        )
        self.dataset_path = dataset_path
        self.dataset = load_dataset("json", data_files=self.dataset_path, split="train")
        self.question_dataset = load_dataset("json", data_files=question_dataset_path, split="train")

    def demo_level_test(self):
        """
        Increase the demo or decrease the demo.
        """
        
        print(f"----------Process the {self.dataset_path}.----------")
        

    def token_level_test(self,):
        """"""

    def recommendation_test(self,):
        """"""
    

    def product_recommendation_test_result(self,demo_dataset_path, ):
        """"""