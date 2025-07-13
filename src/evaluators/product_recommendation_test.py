from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
from datasets import load_dataset
from llmlingua import PromptCompressor
from src.evaluators.inference import qwen3_inference, llama3_inference, phi4_inference, deepseekr1_inference, mistral2_inference
class Product_recommendation():

    def __init__(self, compression_model_name, dataset_path, question_dataset_path, device="cuda:0"):
        self.compression_model_name = compression_model_name
        # self.large_model_name = large_model_name
        self.compression_model = PromptCompressor(
            model_name=compression_model_name,
            device_map=device,
        )
        self.dataset_path = dataset_path
        self.dataset = load_dataset("json", data_files=self.dataset_path, split="train")
        self.question_dataset = load_dataset("json", data_files=question_dataset_path, split="train")
        if "increase" in dataset_path:
            self.flag = "increase"
        elif "decrease" in dataset_path:
            self.flag = "decrease"

    def demo_level_test(self):
        """
        Increase the demo or decrease the demo.
        """
        
        print(f"----------Process the {self.dataset_path}.----------")
        inference_list = [qwen3_inference, llama3_inference, phi4_inference, mistral2_inference]

        for function in inference_list:
            name = qwen3_inference.__name__
            print("-"*10 + f"Inference with {name}!" + "-"*10)
            for compressed in [True, False]:
                function(
                    dataset=self.dataset,
                    question_dataset=self.question_dataset,
                    # large_model_name=self.large_model_name,
                    compression_model=self.compression_model,
                    flag=self.flag,
                    output_path="/home/lzs/Comattack/src/data",
                    compressed=compressed,
                )
            print("-"*10 + f"Finish inference with {name}!" + "-"*10)
            # print("-"*10 + f"Finish inference with {name}!" + "-"*10)

    def token_level_test(self,):
        """"""

    def recommendation_test(self,):
        """"""
    

    def product_recommendation_test_result(self,demo_dataset_path, ):
        """"""


if __name__ == "__main__":
    
    dataset_path = "/home/lzs/Comattack/src/data/replaced_ppl_prep_context_decrease.json"
    