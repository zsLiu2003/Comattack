import nltk.downloader
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import json
import csv
import pandas as pd
from torch import nn
import nltk
from nltk.corpus import wordnet
from datasets import Dataset, load_dataset
from src.utils.get_ppl import get_mean_PPL

# num_steps = 1000

# confuse the compressed model, get the token with high PPL and then select the top-5 or top-10 tokens
# 1. we repeat these top-5 or top-10 tokens to improve the mean PPL of the whole demo. this is the demo level
# 2. we should have a verification to whether the token with high PPL is keyword, the high ppl words are not keyword, so we can edit the keyword.
# 3. edit the token with high PPL to reduce its PPL, resulting in the lower mean PPL of the whole demo
# 4. extract keywords
# 5. edit the keywords to make it has a low PPL so as to remove it.
# 6. change the recommendation result. in other word, we improve the PPL of keywords, in other words, we edit it.

class EditPrompt():

    def __init__(self, data, keywords, model_name, high_ppl_tokens, low_ppl_tokens=None):
        self.data = data
        self.keywords = keywords
        self.high_ppl_tokens = high_ppl_tokens
        self.low_ppl_tokens = low_ppl_tokens
        self.model_name = model_name
    
    def load_dataset(self):
        print("-----------Downloading WordNet data...")
        nltk.download("wordnet")
        nltk.download('omw-1.4')
    

    # def get_ppl(self) -> float:
        
    #     get_mean_PPL(
    #         compression_model_path=""
    #     )
        


    def replace_high_PPL_tokens_in_demo(self):
        # replace the top5 or top10 tokens with high PPL in the demo
        # the tokens with higher PPL are not keywords
        # there is a question. how to replace these high PPL tokens?
        # 1. select some synonym tokens with a smooth and familmiar meaning.
        # 2. providing some paraphrase tokens before the high PPL tokens.
        # 3. add or remove some sudden connection words before the high PPL tokens.
        self.load_dataset()
        
        # device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = GPT2LMHeadModel.from_pretrained(self.model_name, device_map='auto')
        tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
        model.eval()
        device = model.devices
        data = load_dataset("json", data_files=self.high_ppl_tokens, split="train")
        
        print(f"-------------The model is in the {device}------------------")
        print(f"-------------Model name: {self.model_name}")
        


    def replace_low_PPL_tokens_in_demo(self):
        """"""

    def get_high_PPL_tokens(self):
        """"""
    


    def get_low_PPL_tokens(self):
        """"""
# the objective is to effect the recommandation
    def get_insert_tokens(self,):
        """"""

    def get_remove_tokens():
        """"""

    def objective_function():
        """"""
        # first: edit high PPL and key information token to make the compressed model remove them
        # the high PPL token is not the Sufficient and Necessary Condition of key token


        # second: edit the low PPL token to imporve its PPL and make compression model leave them
        # but these tokens is not the key information when recommandation

        # third: insert tokens, we can insert tokens with a high PPL with key information 
        # this is to confuse the recommand LLM

        # forth: remove tokens, remove high PPL tokens without key information
        # after that, the demo seems to maintain a high quality, but it will not be maintained after compression


    def get_edit_tokens():
        """"""