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
from src.data.data_process import get_keyword_dataset, get_target_demo_dataset

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

    def __init__(self, dataset, model_name, phrase_model_name):
        self.dataset = dataset
        # self.keywords = keywords
        # self.high_ppl_tokens = high_ppl_tokens
        # self.low_ppl_tokens = low_ppl_tokens
        self.model_name = model_name
        self.phrase_model_name =  phrase_model_name

    def load_dataset(self):
        print("-----------Downloading WordNet data...")
        nltk.download("wordnet")
        nltk.download('omw-1.4')
        nltk.download('averaged_perceptron_tagger')

    # def get_ppl(self) -> float:
        
    #     get_mean_PPL(
    #         compression_model_path=""
    #     )

    def get_ppl(self, text: str, model:None, tokenizer: None) -> float:
        """
        Calculates the overall perplexity of a given text.
        """

        device = model.device
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

    # 1. In the demo level

    def find_high_and_low_ppl_words(self, sentence: str, top_k: int, model: None, tokenizer: None, flag: bool):
        """
        Analyzes a sentence to find the words with the highest and lowest individual perplexity.
        This helps automatically identify which words to target for optimization.

        Returns:
            A list of tuples, where each tuple is (word, word_perplexity).
        """

        device = model.device
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


    def optimize_with_synonyms(self, model: None, tokenizer: None, phrase_model: None, phrase_tokenizer: None, sentence: str, target_word: str, flag: bool):
        """
        seltect the best synonyms to meet the ppl requirements
        """
        original_ppl = self.get_ppl(
            model=model,
            tokenizer=tokenizer,
            text=sentence,
        )
        sysnonyms = set()
        
        device = model.device
        
        for syn in wordnet.synsets(target_word):
            for lemma in syn.lemmas():
                sysnonym = lemma.name().replace('_', ' ')
                if sysnonym.lower() != target_word.lower() and ' ' not in sysnonym:
                    sysnonyms.add(sysnonym)
        if not sysnonyms: return sentence
        
        best_ppl = original_ppl
        best_sentence = sentence

        for synonym in sysnonyms:
            candidate_sentence = sentence.replace(target_word, synonym)
            candidate_ppl = self.get_ppl(
                text=candidate_sentence,
                model=model,
                tokenizer=tokenizer,
                # device=device,
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

    def get_phrase_context(
            self,
            model: None,
            tokenizer: None,
            prompt: str,
    ):
        """
        To get the proper connectors and pre-context experssion of the given word.
        """  

        device = model.device
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids = inputs.input_ids,
                max_new_tokens=50,
                num_return_sequences=1,
                do_sample=True,
                top_k = 50,
                top_p=0.95,
                pad_token_id = tokenizer.eos_token_id,
            )
        
        input_length = len(inputs.input_ids[0])
        output_text = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
        output_text = output_text.strip()
        phrase_words = [p.strip() for p in output_text.split(",") if p.strip()]

        return phrase_words
                    

    def optimize_with_connectors(self, model: None, tokenizer: None, phrase_model: None, phrase_tokenizer: None, sentence: str, target_word: str, flag: bool):

        """
        Optimize the prompt with connectors
        """
        device = model.device
        original_ppl = self.get_ppl(
            text=sentence,
            model=model,
            tokenizer=tokenizer,
            # device=device,
        )

        parts = sentence.split(f" {target_word}", 1)
        if len(parts) != 2:
            print("-"*10 + "Could not split the sentence by target word" + "-"*10)
            return sentence, original_ppl
        
        pre_target, post_target = parts[0], target_word + " " + parts[1]
        best_ppl = original_ppl
        best_sentence = sentence

        # get the connectors
        # prompt = f"List five different connection words which are suitable be inserted between '{pre_target}' and '{post_target}'. Separate them with commas. Don't output any other content!!!"
        prompt = f"Given the sentence '{sentence}', list five short, common connecting words (like 'so', 'therefore', 'and as a result') that could naturally come before the word '{target_word}'. Separate them with commas.  Don't output any other content!!!"
        replaced_list = self.get_phrase_context(
            model=phrase_model,
            tokenizer=phrase_tokenizer,
            prompt=prompt,
        )
        
        for connector in replaced_list:
            candidate_sentence = f"{pre_target} {connector} {post_target.strip()}"
            candidate_ppl = self.get_ppl(
                text=candidate_sentence,
                model=model,
                tokenizer=tokenizer,    
            )
        
            if candidate_ppl < best_ppl:
                best_ppl = candidate_ppl
                best_sentence = candidate_sentence
            # else:
            #     if candidate_ppl > best_ppl:
            #         best_ppl = candidate_ppl
            #         best_sentence = candidate_sentence
        
        return best_sentence, best_ppl

    def optimize_with_prep_context(self, sentence: str, target_word: str, model: None, tokenizer: None, phrase_model: None, phrase_tokenizer: None, flag: bool):
        """
        Add some prepared context experssion before the high PPL tokens to lower down its PPL
        """
        
        device = model.device
        original_ppl = self.get_ppl(
            text=sentence,
            model=model,
            tokenizer=tokenizer,
            # device=device,
        )
        best_ppl = original_ppl
        best_sentence = sentence


        parts = sentence.split(f" {target_word}", 1)
        if len(parts) != 2:
            print("-"*10 + "Could not split the sentence by target word" + "-"*10)
            return sentence, original_ppl
        
        pre_target = parts[0]
        post_target = target_word + " " + parts[1]

        # prompt = f"List five different prep context to explain"
        prompt = f"Given the text '{sentence}', list three short, common phrases that could naturally come before the word '{target_word}' to make it sound more natural. Phrases should be separated by commas. For example: (in other words, that is to say, to be more specific). Don't output any other content!!!"
        replaced_list = self.get_phrase_context(
            model=phrase_model,
            tokenizer=phrase_tokenizer,
            prompt=prompt,
        )

        for context in replaced_list:
            separator = " " if context and context[-1] not in ".!?" else " "
            # separator = " " if context and context[-1] not in ".!?" else " "
            # candidate_sentence = parts[0] + context + separator + sentence
            candidate_sentence = f"{pre_target} {context} {post_target.strip()}"
            candidate_ppl = self.get_ppl(
                text=candidate_sentence,
                model=model,
                tokenizer=tokenizer,
                # device=device,
            )
            if candidate_ppl < best_ppl:
                best_ppl = candidate_ppl
                best_sentence = candidate_sentence
            
        return best_sentence,best_ppl


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

    # def get_high_PPL_tokens(self):
        # """"""
    
    # def get_low_PPL_tokens(self):
        # """"""


    def decrease_ppl_in_demo(self, model: None, tokenizer: None, phrase_model: None, phrase_tokenizer: None, flag: bool, top_k: int, output_path: str, strategy: str):
        # replace the top5 or top10 tokens with high and low PPL in the demo
        # the tokens with higher PPL are not keywords
        # there is a question. how to replace these high PPL tokens?
        # 1. select some synonym tokens with a smooth and familmiar meaning.
        # 2. providing some paraphrase tokens before the high PPL tokens.
        # 3. add or remove some sudden connection words before the high PPL tokens.
        # self.load_dataset()
        
        # device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # model = GPT2LMHeadModel.from_pretrained(self.model_name, device_map='auto')
        # tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
        # model.eval()
        device = model.device
        # dataset = load_dataset("json", data_files=self.high_ppl_tokens, split="train")

        # print(f"-------------The model is in the {device}------------------")
        # print(f"-------------Model name: {self.model_name}")  

        # print("\n" + "-"*20 + "Automated optimization" + "-"*20)
        
        ppl_list = []
        # sentence_list = []
        for data in tqdm(self.dataset):
            ppl_dict = {}
            sentence_dict = {}
            for key, value in data.items():
                original_ppl = self.get_ppl(
                    text=value,
                    model=model,
                    tokenizer=tokenizer,
                    # device=device
                )
                selected_words = self.find_high_and_low_ppl_words(
                    sentence=value,
                    top_k=top_k,
                    model=model,
                    tokenizer=tokenizer,
                    # device=device,
                    flag=flag,
                )


                # select the correcr functions
                replaced_list = []
                selected_function = None
                prompt = ""
                if strategy == "synonym":
                    selected_function=self.optimize_with_synonyms
                    print("-"*20 + "Decrease the ppl of demo with synonym strategy." + "-"*20)
                elif strategy == "connectors":
                    selected_function=self.optimize_with_connectors
                    print("-"*20 + "Decrease the ppl of demo with connectors strategy." + "-"*20)
                    # replaced_list = conntectors_list
                elif strategy == "prep_context":
                    selected_function=self.optimize_with_prep_context
                    print("-"*20 + "Decrease the ppl of demo with prep_context strategy." + "-"*20)
                    # replaced_list = pre_context_list
                
                if not selected_words:
                    print("-"*10 + "Could not identify any specific ppl word" + "-"*10)
                optimized_sentence = value
                for word_to_replace, _ in selected_words:
                    # 1. decrease the ppl of selected tokens by replacing some synonyms
                    optimized_sentence = selected_function(
                        model=model,
                        tokenizer=tokenizer,
                        phrase_model=phrase_model,
                        phrase_tokenizer=phrase_tokenizer,
                        # device=device,
                        sentence=optimized_sentence,
                        # ppl_of_sentence=
                        # replaced_list=replaced_list,
                        target_word=word_to_replace,
                        flag=flag,
                    )
                    
                    # 2. decrease the ppl of selected tokens by adding some connectors
                    # optimized_sentence_with_connectors = self.optimize_with_connectors(
                    #     model=model,
                    #     tokenizer=tokenizer,
                    #     sentence=optimized_sentence,
                    #     target_word=word_to_replace,
                    #     replaced_list=conntectors_list,
                    # )

                    # # 3. ........by add some experssion contenxt before the selected tokens
                    # optimized_sentence_with_prepared_context = self.optimize_with_prep_context(
                    #     model=model,
                    #     tokenizer=tokenizer,
                    #     sentence=optimized_sentence,
                    #     target_word=word_to_replace,
                    #     pre_contexts_list=pre_context_list,
                    # )
                final_ppl = original_ppl
                if optimized_sentence != value:
                    final_ppl = self.get_ppl(
                        text=optimized_sentence,
                        model=model,
                        tokenizer=tokenizer,
                        # device=device,
                    )
                
                temp_dict = {}
                temp_dict["original"] = value
                temp_dict["replaced"] = optimized_sentence 
                temp_dict["ppl"] = original_ppl
                temp_dict["ppl_replaced"] = final_ppl
                ppl_dict[key] = temp_dict
            ppl_list.append(ppl_dict)

        output_path = f"{output_path}/replaced_ppl_{strategy}_decrease.json"
        with open(output_path,'w', encoding='utf-8') as file:
            json.dump(ppl_list, file, indent=4)

    def optimize_with_adjectives(
            self,
            model: None,
            tokenizer: None,
            phrase_model: None,
            phrase_tokenizer: None,
            sentence: str,
            target_word: str,
            # replaced_list: list,
            flag: bool,
    ):
        original_ppl = self.get_ppl(
            text=sentence,
            model=model,
            tokenizer=tokenizer,
        )
        tagged_words = nltk.pos_tag(sentence.split())
        target_pos = None
        for word, pos in tagged_words:
            if word.strip(".,!?") == target_word:
                target_pos = pos
                break

        if not target_pos:
            print("-"*10 + "Could not determine the part-of-speech for this target word." + "-"*10)
            return sentence, original_ppl
        
        # generate appropriate adjective or adverb word for target_word
        type = ""
        if target_pos.startswith("N"): # Noun (NN, NNS, NNP, NNPS)
            type = "adjective"
        elif target_pos.startswith("V"): # Verb (VB, VBD...)
            type = "adverb"
        else:
            print(f"----------{target_word} is not a Noun or Verb, Skipping.----------")
            return sentence, original_ppl
        
        prompt = f"List three creative, unusual {type}s to describe the word '{target_word}'. Separate them with commas. Don't output any other content!!!"
        added_words = self.get_phrase_context(
            model=phrase_model,
            tokenizer=phrase_tokenizer,
            prompt=prompt,
        )

        best_ppl = original_ppl
        best_sentence = sentence
        for added_word in added_words:
            if type == "adjective":
                candidate_sentence = sentence.replace(target_word, f"{added_word} {target_word}")
            else:
                candidate_sentence = sentence.replace(target_word, f"{target_word} {added_word}")

            candidate_ppl = self.get_ppl(
                text=candidate_sentence,
                model=model,
                tokenizer=tokenizer,    
            )
            if candidate_ppl > best_ppl:
                best_ppl = candidate_ppl
                best_sentence = candidate_sentence

        return best_sentence, best_ppl


    def increase_ppl_in_demo(self, model: None, tokenizer: None, phrase_model: None, phrase_tokenizer: None, flag: bool, top_k: int, output_path: str, strategy: str):
        """
        Decrease the ppl of selected tokens with low PPL
        1. replace the low PPL tokens with their synonyms;
        2. add an adjective before a noun or add a adverd before the verd
        """
        # self.load_dataset()
        
        # device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # model = GPT2LMHeadModel.from_pretrained(self.model_name, device_map='auto')
        # tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
        # model.eval()
        device = model.device
        # dataset = load_dataset("json", data_files=self.high_ppl_tokens, split="train")

        print("-"*20 + "Automated optimize demo by increase its mean PPL." + "-"*20)
        
        # select the proper strategy function
        selected_function = None
        if strategy == "synonym":
            selected_function = self.optimize_with_synonyms
            print("-"*20 + "Increase the ppl of demo with synonym strategy." + "-"*20)
        elif strategy == "adjective":
            selected_function =  self.optimize_with_adjectives
            print("-"*20 + "Increase the ppl of demo with adjectives and adverbs." + "-"*20)

        ppl_list = []        
        for data in tqdm(self.dataset):
            ppl_dict = {}
            for key, value in data.items():
                original_ppl = self.get_ppl(
                    text=value,
                    model=model,
                    tokenizer=tokenizer,
                )
                selected_words = self.find_high_and_low_ppl_words(
                    sentence=value,
                    top_k=top_k,
                    model=model,
                    tokenizer=tokenizer,
                    # device=device,
                    flag=False
                )
                
                optimized_sentence = value
                for target_word, _ in selected_words:
                    optimized_sentence = selected_function(
                        model=model,
                        tokenizer=tokenizer,
                        phrase_model=phrase_model,
                        phrase_tokenizer=phrase_tokenizer,
                        sentence=optimized_sentence,
                        target_word=target_word,
                        flag=flag,
                        # replaced_list=None,
                    )
                    new_ppl = original_ppl
                    if optimized_sentence != value:
                        new_ppl = self.get_ppl(
                            text=optimized_sentence,
                            model=model,
                            tokenizer=tokenizer,
                        )
                temp_dict = {}
                temp_dict["original"] = value
                temp_dict["replaced"] = optimized_sentence 
                temp_dict["ppl"] = original_ppl
                temp_dict["ppl_replaced"] = new_ppl
                ppl_dict[key] = temp_dict
            ppl_list.append(ppl_dict)
        
        output_path = f"{output_path}/replaced_ppl_{strategy}_increase.json"
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(ppl_list, file, indent=4)

# 2. in the token/word level, all the functions above is in the demo level, the following code is for the word level

    def get_keyword_ppl_in_context(self, sentence: str, keyword: str, model: None, tokenizer: None):
        
        """
        NEW: Only calculate the PPL of one word,
        return (keyword_ppl, full_sentence_ppl)
        """
        
        device = model.device
        try:
            keyword_start_char = sentence.index(keyword)
            keyword_end_char = keyword_start_char + len(keyword)
        except ValueError:
            return float('inf'), float('inf')

        # Tokenize with offset mapping to find keyword tokens
        encodings = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True)
        input_ids = encodings.input_ids.to(device)
        offset_mapping = encodings.offset_mapping[0]

        keyword_token_indices = []
        for i, offset in enumerate(offset_mapping):
            start, end = offset.tolist()
            if start >= keyword_start_char and end <= keyword_end_char:
                keyword_token_indices.append(i)

        if not keyword_token_indices:
            # print(f"Warning: Could not map keyword '{keyword}' to tokens.")
            return float('inf'), float('inf')

        # Get per-token loss for the whole sentence
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        keyword_token_losses = []
        for token_idx in keyword_token_indices:
            if token_idx > 0: # The first token has no loss
                keyword_token_losses.append(per_token_loss[token_idx - 1].item())

        if not keyword_token_losses:
            return float('inf'), torch.exp(outputs.loss).item()

        keyword_ppl = np.exp(np.mean(keyword_token_losses))
        sentence_ppl = torch.exp(outputs.loss).item()
        
        return keyword_ppl, sentence_ppl


    def optimize_with_character_edits(
        self,
        model: None,
        tokenizer: None,
        sentence: str,
        target_word: str,
    ):
        """
        Insert, delete, or replace one character in one word to decrease the PPL of keyword.
        Return (optimized sentence, optimized_word, optimized_keyword_ppl)
        """
        
        original_keyword_ppl, _ = self.get_keyword_ppl_in_context(
            model=model,
            tokenizer=tokenizer,
            sentence=sentence,
            keyword=target_word,
        )
        if original_keyword_ppl == float("inf"): 
            return sentence, target_word, original_keyword_ppl
        
        # find the delete, insert, and replaced characters
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(target_word[:i], target_word[i:]) for i in range(len(target_word) + 1)]
        deletes    = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts    = [L + c + R for L, R in splits for c in letters]

        candidates = set(deletes + transposes + replaces + inserts)
        valid_candidates = {word for word in candidates if wordnet.synsets(word)}
        if not valid_candidates:
            print("-"*10 + "No valid single-edit word found." + "-"*10)
            return sentence, target_word, original_keyword_ppl
        
        best_keyword_ppl = original_keyword_ppl
        best_sentence = sentence
        best_keyword = target_word

        for word in valid_candidates:
            candidate_sentence = sentence.replace(target_word, word)
            candidata_keyword_ppl, _ = self.get_keyword_ppl_in_context(
                model=model,
                tokenizer=tokenizer,
                sentence=candidate_sentence,
                keyword=word,
            )
            if candidata_keyword_ppl < best_keyword_ppl:
                best_keyword_ppl = candidata_keyword_ppl
                best_sentence = candidate_sentence
                best_keyword = word
        
        return best_sentence, best_keyword, best_keyword_ppl
    
    def optimizer_with_symbol(
            self,
            model: None,
            tokenizer: None,
            sentence: str,
            target_word: str,
    ):
        """
        Decrease the PPL of the keyword by insert some symbols
        return (optimized_sentence, optimized_keyword, optimized_keyword_ppl)
        """

        original_keyword_ppl, _ = self.get_keyword_ppl_in_context(
            model=model,
            tokenizer=tokenizer,
            sentence=sentence,
            keyword=target_word,
        )
        if original_keyword_ppl == float("inf"):
            return sentence, target_word, original_keyword_ppl
        
        framing_patterns = [f'"{target_word}"', f"'{target_word}'", f"{target_word},"]
        if ' ' in target_word:
            framing_patterns.append(target_word.replace(' ', '-'))

        best_keyword = target_word
        best_keyword_ppl = original_keyword_ppl
        best_sentence = sentence
        
        for pattern in framing_patterns:
            candidate_sentence = sentence.replace(target_word, pattern)
            candidate_keyword_ppl, _ = self.get_keyword_ppl_in_context(
                model=model,
                tokenizer=tokenizer,
                sentence=candidate_sentence,
                keyword=pattern,
            )
            if candidate_keyword_ppl < best_keyword_ppl:
                best_keyword = pattern
                best_sentence = candidate_sentence
                best_keyword_ppl = candidate_keyword_ppl
            
        return best_sentence, best_keyword, best_keyword_ppl
            

    def optimize_with_token_manipulation(
        self, 
        model: None,
        tokenizer: None,
        sentence: str,
        target_word: str,
    ):
        """
        Delelte one token when the length of keyword excceed two token
        return optimized_sentence, optimized_keyword, optimized_keyword_ppl
        """

        original_keyword_ppl, _ = self.get_keyword_ppl_in_context(
            model=model,
            tokenizer=tokenizer,
            sentence=sentence,
            keyword=target_word,
        )
        if original_keyword_ppl == float("inf"):
            return sentence, target_word, original_keyword_ppl
        
        tokens = target_word.split()
        if len(target_word) < 2: 
            return sentence, target_word, original_keyword_ppl
        
        best_sentence = sentence
        bset_keyword = target_word
        best_keyword_ppl = original_keyword_ppl
        
        for i in range(len(tokens)):
            new_keyword = ' '.join(tokens[:i] + tokens[i+1:])
            if not new_keyword: continue
            candidate_sentence = sentence.replace(target_word, new_keyword)
            candidate_keyword_ppl, _ = self.get_keyword_ppl_in_context(
                model=model,
                tokenizer=tokenizer,
                sentence=candidate_sentence,
                keyword=new_keyword
            )
            if candidate_keyword_ppl < best_keyword_ppl:
                best_keyword_ppl, best_sentence, best_keyword = candidate_keyword_ppl, candidate_sentence, new_keyword
    
        return best_sentence, best_keyword, best_keyword_ppl

    # execution function of word/token level edit
    def optimize_to_ppl_threshold(
        self, 
        model: None, 
        tokenizer: None, 
        keyword_dataset: None,
        top_k: int, 
        k: int,
        output_path: str,
        # strategy: str,
    ):

        """
        Decrease the ppl of keyword to one threshold
        """
        
        # selected_function = None
        # if strategy == "character":
        #     selected_function = self.optimize_with_character_edits
        # elif strategy == "symbol":
        #     selected_function = self.optimizer_with_symbol
        # else:
        #     selected_function = self.optimize_with_token_manipulation


        output_list = []
        for data,keywrods in tqdm(zip(self.dataset,keyword_dataset)):
            output_dict = {}
            for key, value in data.items():
                keyword = keywrods[key]
                word_list = self.find_high_and_low_ppl_words(
                    model=model,
                    tokenizer=tokenizer,
                    sentence=value,
                    top_k=top_k,
                    flag=False,
                )
                word, threshold_ppl = word_list[-k]
                # candidate_sentence = value
                current_sentence = value
                # best_keyword = target_word
                for target_word in keyword:
                    if target_word not in value:
                        break
                    # candidate_sentence,candidate_keyword, candidate_keyword_ppl = selected_function(
                    #     model=model,
                    #     tokenizer=tokenizer,
                    #     sentence=candidate_sentence,
                    #     target_word=target_word,
                    # )
                    _, original_word_ppl = self.get_keyword_ppl_in_context(
                        model=model,
                        tokenizer=tokenizer,
                        sentence=value,
                        keyword=target_word,
                    )
                    best_sentence_for_this_word, best_keyword, best_ppl = current_sentence, target_word, original_word_ppl

                    # 1. optimized with character_edit
                    char_s, char_k, char_p = self.optimize_with_character_edits(
                            model=model,
                            tokenizer=tokenizer,
                            sentence=value,
                            target_word=target_word,
                        )
                    if char_p < best_ppl: best_sentence_for_this_word, best_keyword, best_ppl = char_s, char_k, char_p

                    # 2. optimized with added symbol
                    sym_s, sym_k, sym_p = self.optimizer_with_symbol(
                            model=model,
                            tokenizer=tokenizer,
                            sentence=best_sentence_for_this_word,
                            target_word=best_keyword,
                        )
                    if sym_p < best_ppl: best_sentence_for_this_word, best_keyword, best_ppl = sym_s, sym_k, sym_p   
                    
                    # 3. optimized with token deletion
                    # if ' ' in best_keyword:
                    tok_s, tok_k, tok_p = self.optimize_with_token_manipulation(
                        model=model,
                        tokenizer=tokenizer,
                        sentence=best_sentence_for_this_word,
                        target_word=best_keyword,
                    )
                    if tok_p < best_ppl: best_sentence_for_this_word, best_keyword, best_ppl = tok_s, tok_k, tok_p
                    # else:
                     
                        
                    # 4. Chekc if the optimized ppl < threshold_ppl
                    if best_ppl < threshold_ppl:
                        # print(f"   SUCCESS: Optimized '{word_to_fix}' -> '{best_keyword}'. New PPL {best_ppl:.4f} is below threshold.")
                        print("-"*10 + "Successfully" + "-"*10)
                        temp_dict = {}
                        temp_dict["original"] = value
                        temp_dict["replaced"] = best_sentence_for_this_word
                        temp_dict["original_keyword"] = target_word
                        temp_dict["replaced_keyword"] = best_keyword
                        temp_dict["original_keyword_ppl"] = original_word_ppl
                        temp_dict["replaced_keyword_ppl"] = best_ppl
                        temp_dict["threshold_ppl"] = threshold_ppl

                        output_dict[key] = temp_dict

                        # output_dict[key] = best_sentence_for_this_word
                        
                        # current_sentence = best_sentence_for_this_word
                        
                        break
                    current_sentence = best_sentence_for_this_word
                    # else:
                        # print(f"   FAILURE: Could not optimize '{word_to_fix}' to be below the threshold. Best PPL found was {best_ppl:.4f}.")
            
            output_list.append(output_dict)
            output_path = f"{output_path}/decrease_keyword_ppl.json"
            with open(output_path, "w", encoding='utf-8') as file:
                json.dump(output_list, file, indent=4)


# 3. in the recommendation level, the objective is to effect the recommandation

    def get_confused_words(self):
        """
        Get the word list to increase or decrease the probability of remmendation
        return positive_words, superlative_words, trust_words, weakning_words, hesitant_words, uncertainty_words
        """

        # increase
        positive_words = ["revolutionary", "groundbreaking", "unparalleled", "state-of-the-art", "essential"]
        superlative_words = {
            "good": ["outstanding", "superior", "excellent", "exceptional"],
            "fast": ["blazing-fast", "instantaneous", "ultra-responsive"],
            "powerful": ["robust", "high-performance", "dominant"]
        }
        trust_words = ["expert-approved", "industry-standard", "certified", "award-winning"]

        # decrease
        weakning_words = ["decent", "adequate", "suitable", "workable"]
        hesitant_words = [
            "While it's a solid option,",
            "It's worth considering that",
            "One perspective is that",
            "Although effective in some cases,"
        ]
        uncertainty_words = {
            "is": ["can be", "is often", "is sometimes"],
            "will": ["may", "could potentially", "might"]
        }

        return positive_words, superlative_words, trust_words, weakning_words, hesitant_words, uncertainty_words
    
    def recommendation_manipulation(
        self, 
        # model: None, 
        # tokenizer: None, 
        target_dataset: None,
    ):
        """
        Edit the description of one product or one tool.
        
        Args:
            target_dataset: the index of the best demo and the target demo.
        
        Return:
        """
        import random

        positive_words, superlative_words, trust_words, weakning_words, hesitant_words, uncertainty_words = self.get_confused_words()
        
        output_list = []
        for data,index in tqdm(zip(self.dataset, target_dataset)):
            best_index = index["best"]
            target_index = index["target"]
            output_dict = {}
            for key,value in data.items():
                if key == target_index:
                    # Strategy 1: Prepend a strong positive keyword and a trust trigger
                    new_demo = f"{random.choice(trust_words).capitalize()}, this {random.choice(positive_words)} product {value.lower()}"
                    for word, replacements in superlative_words.items():
                        if word in new_demo:
                            new_demo = new_demo.replace(word, random.choice(replacements), 1)
                    temp_dict = {}
                    temp_dict["original"] = value
                    temp_dict["new"] = new_demo
                    output_dict[key] = temp_dict
                else:
                    #  --- Strategy 2: Weaken the Competitors ---
                    new_demo = value
                    for word in superlative_words.keys():
                        if word in new_demo:
                            new_demo = new_demo.replace(word, random.choice(weakning_words), 1)
                    
                    for word, replacements in uncertainty_words.items():
                        if re.search(r'\b' + word + r'\b' + new_demo):
                            new_demo = re.sub(r'\b' + word + r'\b', random.choice(replacements), new_demo, 1)
                            break
                    
                    new_demo = f"{random.choice(hesitant_words)} {new_demo.lower()}"
                    temp_dict = {}
                    temp_dict["original"] = value
                    temp_dict["new"] = new_demo
                    output_dict[key] = temp_dict
            
            output_list.append(output_dict)
            output_path = f"{output_path}/confused_recommendation.json"
            
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(output_list, file, indent=4)


    # def get_insert_tokens(self,):
    #     """"""

    # def get_remove_tokens(self,):
    #     """"""

    # def objective_function(self,):
    #     """"""
    #     # first: edit high PPL and key information token to make the compressed model remove them
    #     # the high PPL token is not the Sufficient and Necessary Condition of key token


    #     # second: edit the low PPL token to imporve its PPL and make compression model leave them
    #     # but these tokens is not the key information when recommandation

    #     # third: insert tokens, we can insert tokens with a high PPL with key information 
    #     # this is to confuse the recommand LLM

    #     # forth: remove tokens, remove high PPL tokens without key information
    #     # after that, the demo seems to maintain a high quality, but it will not be maintained after compression


    def get_edit_tokens(
        self,
        keywords_dataset_path: str,
        target_demo_path: str,
        output_path: str,
        top_k: int,
    ):
        """"""
        
        # connector_list = ["so", "and", "therefore", "as a result", "consequently", "because", "however", "in addition", "what's more", "else", "hence"]
        # pre_context_list = [
        #     "for example,",
        #     "specifically,",
        #     "that is to say,"
        # ]

        # initial compression model
        model = GPT2LMHeadModel.from_pretrained(self.model_name, device_map='auto')
        tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
        model.eval()
        # device = model.device
        # initial inference model
        phrase_model = AutoModelForCausalLM.from_pretrained(self.phrase_model_name, device_map="auto")
        phrase_tokenizer = AutoTokenizer.from_pretrained(self.phrase_model_name)
        phrase_model.eval()

        # dataset = load_dataset("json", data_files=self.high_ppl_tokens, split="train")
        # get dataset
        self.load_dataset()

        keyword_dataset = get_keyword_dataset(dataset_path=keywords_dataset_path)
        target_demo_dataset = get_target_demo_dataset(dataset_path=target_demo_path)

        # 1. demo level edit
        # 1.a decrease the mean ppl of the whole demo
        decrease_strategy_list = ["synonym", "connectors", "prep_context"]
        increase_strategy_list = ["sysnonym", "adjectives"]
        
        for strategy in decrease_strategy_list:
            self.decrease_ppl_in_demo(
                model=model,
                tokenizer=tokenizer,
                phrase_model=phrase_model,
                phrase_tokenizer=phrase_tokenizer,
                flag=False,
                top_k=20,
                output_path=output_path,
                # conntectors_list=,
                # pre_context_list=,
                strategy=strategy,
            )
        
        print("-"*20 + "Successfully finishing that task: Decrease the ppl of demo." + "-"*20)
        
        # 1.b increase the mean ppl of the whole demo
        for strategy in increase_strategy_list:
            self.increase_ppl_in_demo(
                model=model,
                tokenizer=tokenizer,
                phrase_model=phrase_model,
                phrase_tokenizer=phrase_tokenizer,
                flag=False,
                top_k=20,
                output_path=output_path,
                strategy=strategy,
            )
        print("-"*20 + "Successfully finishing that task: Increase the ppl of demo." + "-"*20)

        # 2. token or word level edit, including editing the keywords in the demo
        self.optimize_to_ppl_threshold(
            model=model,
            tokenizer=tokenizer,
            keyword_dataset=keyword_dataset,
            top_k=top_k,
            k=int(top_k / 2),
            output_path=output_path,
        )

        print("-"*20 + "Successfully finishing that task: Decrease the ppl of keyword." + "-"*20)

        # 3. how to affect the product recommandation result
        self.recommendation_manipulation(
            target_dataset=target_demo_dataset,
        )
        
        print("-"*20 + "Successfully finishing that task: Manipulate the product recommendation." + "-"*20)

