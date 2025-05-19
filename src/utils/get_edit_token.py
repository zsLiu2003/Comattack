import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import csv
import pandas as pd
from torch import nn

num_steps = 1000

# confuse the compressed model, get the token with high PPL and then select the top-5 or top-10 tokens
# 1. we repeat these top-5 or top-10 tokens to improve the mean PPL of the whole demo.
# 2. we should have a verification to whether the token with high PPL is keyword
# 3. edit the token with high PPL to reduce its PPL, resulting in the lower mean PPL of the whole demo
# 4. extract keywords
# 5. edit the keywords to make it has a low PPL so as to remove it.
# 6. change the recommendation result. in other word, we improve the PPL of keywords, in other words, we edit it.
def get_high_PPL_tokens():
    """"""
    


def get_low_PPL_tokens():
    """"""
# the objective is to effect the recommandation
def get_insert_tokens():
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

def main():
    """"""