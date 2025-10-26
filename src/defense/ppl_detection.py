import torch
import tiktoken
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
model_name = "models/gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_name).eval()
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Your input text
text = "hello world aaaaaaaaaaaa"

# Encode with tiktoken (optional, for debugging or BPE analysis)
enc = tiktoken.get_encoding("gpt2")
bpe_tokens = enc.encode(text)
print("Tiktoken BPE tokens:", bpe_tokens)

# Encode with transformers tokenizer
inputs = tokenizer(text, return_tensors="pt")

# Get model output
with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss  # This is average negative log-likelihood

# Compute perplexity
ppl = torch.exp(loss)
print(f"Perplexity: {ppl.item():.2f}")
