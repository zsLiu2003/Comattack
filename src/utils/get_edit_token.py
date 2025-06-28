import nltk.downloader
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import json
import csv
import pandas as pd
from torch import nn
import re
import numpy as np
import nltk
from nltk.corpus import wordnet
from datasets import Dataset, load_dataset
# from src.utils.get_ppl import get_mean_PPL
# from src.utils.verify_PPL import get_single_PPL
from tqdm import tqdm

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

    def get_ppl(self, text: str, model:None, tokenizer: None, device: str) -> float:
        """
        Calculates the overall perplexity of a given text.
        """
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        
        max_length = model.config.n_positions
        if input_ids.size(1) > max_length:
            input_ids = input_ids[:, :max_length]

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        ppl = torch.exp(loss)
        return ppl.item()

    def find_high_and_low_ppl_words(self, sentence: str, top_k: int, model: None, tokenizer: None, device: str, flag: bool):
        """
        Analyzes a sentence to find the words with the highest and lowest individual perplexity.
        This helps automatically identify which words to target for optimization.

        Returns:
            A list of tuples, where each tuple is (word, word_perplexity).
        """
        # Tokenize the sentence and get the model's predictions (logits)
        encodings = tokenizer(sentence, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            logits = outputs.logits

        # Use CrossEntropyLoss with 'none' reduction to get loss for each token
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        # Shift logits and labels for calculating loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_token_ppl = torch.exp(per_token_loss)

        # Re-align tokens with their PPL scores
        tokens = [tokenizer.decode(token_id) for token_id in input_ids[0]]
        
        # Group subword tokens back into whole words and calculate their average PPL
        word_ppls = []
        current_word = ""
        current_word_losses = []
        
        # Start from the second token since the first token has no loss/PPL
        for i, token_id in enumerate(input_ids[0][1:]):
            token_str = tokenizer.decode(token_id)
            # GPT-2 tokenizer uses 'Ġ' to mark the start of a new word
            if token_str.startswith('Ġ') or i == 0:
                if current_word:
                    word_ppls.append((current_word, np.mean(current_word_losses)))
                current_word = token_str.lstrip('Ġ')
                current_word_losses = [per_token_ppl[i].item()]
            else:
                current_word += token_str
                current_word_losses.append(per_token_ppl[i].item())
        
        # Add the last word
        if current_word:
            word_ppls.append((current_word, np.mean(current_word_losses)))

        # Sort words by their PPL in descending order and return the top N
        word_ppls.sort(key=lambda x: x[1], reverse=True)
        
        if flag:
            return word_ppls[:top_k]

        return word_ppls[-top_k:]


    def optimize_with_synonyms(self, model, tokenizer, device, senetence: str, ppl_of_senetence: float, target_word: str, flag: bool):
        """
        seltect the best synonyms to meet the ppl requirements
        """
        original_ppl = ppl_of_senetence
        sysnonyms = set()
        
        for syn in wordnet.synsets(target_word):
            for lemma in syn.lemmas():
                sysnonym = lemma.name().replace('_', ' ')
                if sysnonym.lower() != target_word.lower() and ' ' not in sysnonym:
                    sysnonyms.add(sysnonym)
        if not sysnonyms: return senetence
        
        best_ppl = original_ppl
        best_sentence = senetence

        for synonym in sysnonyms:
            candidate_sentence = senetence.replace(target_word, synonym)
            candidate_ppl = self.get_ppl(
                text=candidate_sentence,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            if flag:
                if candidate_ppl < best_ppl:
                    best_ppl = candidate_ppl
                    best_sentence = candidate_sentence
            else:
                if candidate_ppl > best_ppl:
                    best_ppl = candidate_ppl
                    best_sentence = candidate_sentence
        return best_sentence


    def replace_tokens_in_demo(self, flag: bool, top_k: int, output_path):
        # replace the top5 or top10 tokens with high and low PPL in the demo
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
        device = model.device
        dataset = load_dataset("json", data_files=self.high_ppl_tokens, split="train")

        print(f"-------------The model is in the {device}------------------")
        print(f"-------------Model name: {self.model_name}")  

        print("\n" + "-"*20 + "Automated optimization" + "-"*20)
        
        ppl_list = []
        sentence_list = []
        for data in tqdm(dataset):
            ppl_dict = {}
            sentence_dict = {}
            for key, value in data.items():
                original_ppl = self.get_ppl(
                    text=value,
                    model=model,
                    tokenizer=tokenizer,
                    device=device
                )
                selected_words = self.find_high_and_low_ppl_words(
                    sentence=value,
                    top_k=top_k,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    flag=flag,
                )
                if not selected_words:
                    print("-"*10 + "Could not identify any specific ppl word" + "-"*10)
                optimized_sentence = value
                for word_to_replace, _ in selected_words:
                    optimized_sentence = self.optimize_with_synonyms(
                        senetence=optimized_sentence,
                        target_word=word_to_replace,
                    )
                if optimized_sentence != value:
                    final_ppl = self.get_ppl(
                        text=optimized_sentence,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                    )
                
                temp_dict = {}
                temp_dict["original"] = value
                temp_dict["replaced"] = optimized_sentence 
                temp_dict["ppl"] = original_ppl
                temp_dict["ppl_replaced"] = final_ppl
                ppl_dict[key] = temp_dict
            ppl_list.append(ppl_dict)

        output_path = f"{output_path}/replaced_ppl.json"
        with open(output_path,'w', encoding='utf-8') as file:
            json.dump(ppl_list, file, indent=4)
                    
                
        
    # def replace_low_PPL_tokens_in_demo(self):
    #     """
    #     Replace the tokens with lower PPL in demo with some synonyms to improve the ppl of the whole demo.
    #     """
    #     self.load_dataset()
    #     model = GPT2LMHeadModel.from_pretrained(self.model_name, device_map='auto')
    #     tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
    #     model.eval()
    #     device = model.devices
    #     data = load_dataset("json", data_files=self.high_ppl_tokens, split="train")

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