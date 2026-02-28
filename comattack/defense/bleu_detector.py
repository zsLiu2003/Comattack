import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import language_tool_python
from nltk.util import ngrams
from collections import Counter
import numpy as np
import math

class AttackDetector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', lt_path='./LanguageTool'):
        # Load GPT-2 for PPL
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.model.eval()
        self.device = device
        # Load grammar checker
        self.tool = language_tool_python.LanguageTool('en-US')

    def compute_perplexity(self, text):
        encodings = self.tokenizer(text, return_tensors="pt").to(self.device)
        max_length = self.model.config.n_positions
        stride = 512

        lls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = i + stride
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                log_likelihood = outputs.loss * target_ids.size(1)

            lls.append(log_likelihood)

        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        return ppl.item()

    def compute_ngram_repetition(self, text, n=4):
        tokens = text.split()
        total_ngrams = list(ngrams(tokens, n))
        num_total = len(total_ngrams)
        num_unique = len(set(total_ngrams))
        if num_total == 0:
            return 0.0
        repetition_rate = 1.0 - (num_unique / num_total)
        return repetition_rate

    def grammar_errors(self, text):
        matches = self.tool.check(text)
        return len(matches)

    def detect(self, texts):
        """
        texts: List[str]  # each is an attacked text
        Returns: List[Dict]
        """
        results = []
        for text in texts:
            ppl = self.compute_perplexity(text)
            rep = self.compute_ngram_repetition(text)
            errors = self.grammar_errors(text)

            results.append({
                'text': text,
                'perplexity': ppl,
                'ngram_repetition_4': rep,
                'grammar_errors': errors
            })
        return results
    
if __name__ == "__main__":
    lt_path = './LanguageTool'
    detector = AttackDetector()
    sample_texts = [
        "CBS broadcast Super Bowl 50 in the U.S., and charged an average of $5 million for a 30-second commercial during the game. The Super Bowl 50 halftime show was headlined by the British rock groupTo⁣ with special guest performers Beyoncé and Errorf Mars, who headlined the Super Bowl XLVII and Super Bowl XLVIII halftime shows, respectively. It was the third-most watched U.S. broadcast ever.",
        
    ]
    results = detector.detect(sample_texts)
    for res in results:
        print(f"Text: {res['text']}\nPPL: {res['perplexity']}, N-gram Repetition: {res['ngram_repetition_4']}, Grammar Errors: {res['grammar_errors']}\n")