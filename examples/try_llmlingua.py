
from llmlingua import PromptCompressor
import torch
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmlingua import PromptCompressor
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import pipeline
import argparse
import numpy as np

def get_PPL(model, tokenizer, origin_text, compressed_text):

    device = model.device
    input = tokenizer(origin_text, return_tensors='pt').to(device)
    input_ids1 = input["input_ids"]
    input_ids2 = tokenizer(compressed_text, return_tensors='pt')["input_ids"]
    
     # get the logits 
    with torch.no_grad():
        output = model(**input)
        logits = output.logits

    # calculate the cross_entropy loss
    shift_logits = logits[...,:-1, :].contiguous()
    label_labels = input_ids1[...,1:].contiguous()
    loss_function = CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
    loss = loss_function(
        shift_logits.view(-1,shift_logits.size(-1)), 
        label_labels.view(-1),
        )
    ppl_per_token = torch.exp(loss).cpu().numpy().tolist()
    ppl_mean_origin = torch.exp(loss).mean().item()

    # select the ppl of compressed_tokens
    le1 = input_ids1.size(1)
    le2 = input_ids2.size(1)
    id2_index = 0
    compressed_ppl = []
    le = min(le1, len(ppl_per_token))
    print(len(ppl_per_token))
    print(le1)
    for id1_index in range(le):
        if id2_index < le2 and input_ids1[0, id1_index] == input_ids2[0, id2_index]:
            compressed_ppl.append(ppl_per_token[id1_index])
            id2_index += 1
    ppl_mean_compressed = np.mean(compressed_ppl)

    return ppl_per_token, ppl_mean_origin, compressed_ppl, ppl_mean_compressed

device = "cuda:2" if torch.cuda.is_available() else "cpu"
model_name = "models/gpt2-dolly"
compressor = PromptCompressor(
    model_name=model_name,
    device_map=device,
    )

prompt = "Angelo and Mselanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?"
GSM8K_PROMPT = "Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\nAngelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n\nQuestion: You can buy 4 apples or 1 watermelon for the same price. You bought 36 fruits evenly split between oranges, apples and watermelons, and the price of 1 orange is $0.50. How much does 1 apple cost if your total bill was $66?\nLet's think step by step\nIf 36 fruits were evenly split between 3 types of fruits, then I bought 36/3 = 12 units of each fruit\nIf 1 orange costs $0.50 then 12 oranges will cost $0.50 * 12 = $6\nIf my total bill was $66 and I spent $6 on oranges then I spent $66 - $6 = $60 on the other 2 fruit types.\nAssuming the price of watermelon is W, and knowing that you can buy 4 apples for the same price and that the price of one apple is A, then 1W=4A\nIf we know we bought 12 watermelons and 12 apples for $60, then we know that $60 = 12W + 12A\nKnowing that 1W=4A, then we can convert the above to $60 = 12(4A) + 12A\n$60 = 48A + 12A\n$60 = 60A\nThen we know the price of one apple (A) is $60/60= $1\nThe answer is 1"

compressed_prompt = compressor.compress_prompt(
    GSM8K_PROMPT[:1000],
    instruction='',
    question='',
    target_token=200,
)
print(len(compressed_prompt["compressed_prompt"]), len(GSM8K_PROMPT))
print(compressed_prompt["origin_tokens"], compressed_prompt["compressed_tokens"])
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ppl_all, ppl_mean, ppl_all_compressed, ppl_mean_compressed = get_PPL(
    model = model,
    tokenizer = tokenizer,
    origin_text = GSM8K_PROMPT[:1000],
    compressed_text = compressed_prompt["compressed_prompt"],
)
print(ppl_mean,ppl_mean_compressed)

