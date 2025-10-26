from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
from datasets import load_dataset
from llmlingua import PromptCompressor
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src.evaluators.inference import qwen3_inference, llama3_inference, phi4_inference, deepseekr1_inference, mistral2_inference, llama2_inference
from src.utils.get_edit_token import EditPrompt



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
        else:
            self.flag = "confused"
        if "keyword" in dataset_path:
            self.flag = "keyword"

    def demo_level_test(self):
        """
        Increase the demo or decrease the demo.
        For the decrease: test whether the degraded demo can be removed.
        For the demo with the lowest ppl: test whether this demo can be leaved
        """
        
        print(f"----------Process the {self.dataset_path}.----------")
        inference_list = [llama2_inference]

        for function in inference_list:
            name = function.__name__
            print("-"*10 + f"Inference with {name}!" + "-"*10)
            output_path = "src/data"
            if "adj" in self.dataset_path:
                output_path = "src/data2"
            # elif "confused" in self.dataset_path:
            #     output_path = "src/data3"
            # elif "keyword" in self.dataset_path:
            #     output_path = "src/data4"
            for compressed in [True, False]:
                function(
                    dataset=self.dataset,
                    question_dataset=self.question_dataset,
                    # large_model_name=self.large_model_name,
                    compression_model=self.compression_model,
                    flag=self.flag,
                    output_path=output_path,
                    compressed=compressed,
                    common=True if "confused" in self.dataset_path or "keyword" in self.dataset_path else False
                )
            print("-"*10 + f"Finish inference with {name}!" + "-"*10)
            # print("-"*10 + f"Finish inference with {name}!" + "-"*10)

    def token_level_test(self, dataset, model_name, phrase_model_name, flag):
        """
        Check the token level test. 
        1. In the demo level experiment, we decrease the ppl of high ppl words, these words will be remove
        2. We increase the ppl of low ppl words, these words will be maintained
        # 3. We degrade the ppl of keywords, these keywords will be removed
        """

        Edit = EditPrompt(
            dataset=dataset,
            model_name=model_name,
            phrase_model_name=phrase_model_name,
        ) 
        
        print(f"----------Process the {dataset}.----------")
        model = GPT2LMHeadModel.from_pretrained(self.compression_model_name, device_map='auto')
        tokenizer = GPT2TokenizerFast.from_pretrained(self.compression_model_name)
        model.eval()
        output_list = []
        le = 20
        result = 0
        dict_num = 0
        for data_entry in tqdm(dataset):
            output_dict = {}
            for key, value in data_entry.items():
                dict_num += 1
                target_ppl_words = Edit.find_high_and_low_ppl_words(
                    sentence=value["original"],
                    top_k=20,
                    model=model,
                    tokenizer=tokenizer,
                    flag=flag
                )
                original_compressed = self.compression_model.compress_prompt(
                    value["original"],
                    instruction="",
                    question="",
                    target_token=le,
                )
                original_compressed = original_compressed["compressed_prompt"]
                optimized_compressed = self.compression_model.compress_prompt(
                    value["replaced"],
                    instruction="",
                    question="",
                    target_token=le,
                )
                optimized_compressed = optimized_compressed["compressed_prompt"]
                # output = False
                for word, _ in target_ppl_words:
                    if word in original_compressed and word not in optimized_compressed:
                        # output = True
                        result += 1
                        break
        
        output = result / dict_num
        print(f"-------------------The result is: {output}-------------------")

        return output
    
    def keywords_test(self, dataset, model_name, phrase_model_name, flag):
        """
        Detect whether the keywords can be removed or maintained.
        """
        # Edit = EditPrompt(
        #     dataset=dataset,
        #     model_name=model_name,
        #     phrase_model_name=phrase_model_name,
        # )
        
        print(f"----------Process the {dataset}.----------")
        model = GPT2LMHeadModel.from_pretrained(self.compression_model_name, device_map='auto')
        tokenizer = GPT2TokenizerFast.from_pretrained(self.compression_model_name)
        model.eval()
        output_list = []
        le = 50
        result = 0
        dict_num = 0
        real_num = 0
        for data_entry in tqdm(dataset):
            # for key, value in data_entry.items():
                
            #     dict_num += 1
            #     original_compressed = self.compression_model.compress_prompt(
            #         value["original"],
            #         instruction="",
            #         question="",
            #         target_token=le,
            #     )
            #     original_compressed = original_compressed["compressed_prompt"]
            #     optimized_compressed = self.compression_model.compress_prompt(
            #         value["replaced"],
            #         instruction="",
            #         question="",
            #         target_token=le,
            #     )
            keyword_list = []
            prompt = ""
            for key, value in data_entry.items():
                keyword_list.append(value["replaced_keyword"])
                dict_num += 1
                prompt += value["replaced"]
            compressed_prompt = self.compression_model.compress_prompt(
                value["replaced"],
                instruction="",
                question="",
                target_token=le,
            )
            optimized_compressed = compressed_prompt["compressed_prompt"]
            if value["original_keyword"] != value["replaced_keyword"]:
                real_num += 1
                for keyword in keyword_list:
                    if keyword not in optimized_compressed:
                        result += 1 


        output = result / real_num
        real_output = result / dict_num
        print(f"-------------------The result is: {output}-------------------")
        print(f"-------------------The real result is: {real_output}-------------------")

    def recommendation_test(self, dataset, model_name, phrase_model_name, flag):
        """
        Check the recommendation test:
        1. First, we can directly removed or maintained our target demo.
        2. Second, we can degrade the preference of LLM to one demo by degrading the ppl of its keywords
        3. Third, we can add adj before noum and adv after verb to confuse the preference manipulation of LLM.
        """
        
        # Edit = EditPrompt(
        #     dataset=dataset,
        #     model_name=model_name,
        #     phrase_model_name=phrase_model_name,
        # )
        
        print(f"----------Process the {dataset}.----------")
        # model = GPT2LMHeadModel.from_pretrained(self.compression_model_name, device_map='auto')
        # tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        # model.eval()
        output_list = []
        le = 20
        result = 0
        dict_num = 0
        real_num = 0
        for data_entry in tqdm(dataset):
            output_dict = {}
            for key, value in data_entry.items():
                output_dict[key] = {
                    "original": value["original"],
                    "replaced": value["new"],
                }
            
            output_list.append(output_dict)
        
        output_path = "src/data/replaced_confused_recommendation.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_list, f, indent=4, ensure_ascii=False)
        

    def product_recommendation_test_result(self,demo_dataset_path):
        """"""
        




