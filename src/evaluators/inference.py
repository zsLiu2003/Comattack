from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import transformers
import torch
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import gc
import random
from src.utils.get_prompt import get_target_prompt, get_pure_target_prompt

def dataset_process(dataset, question_dataset, compression_model, flag, compressed, common):
    
    output_list = []
    selection_function = None
    if flag:
        selection_function = max
    else:
        selection_function = min
    
    i = 1
    for data, question_data in tqdm(zip(dataset,question_dataset)): 
        target_demo = "" 
        if common:
            target_demo = random.choice(list(data.keys()))
        else:
            target_demo = selection_function(data, key=lambda k: data[k]["ppl"])
        question = question_data["question"]
        requirements = question_data["requirements"]
        system_prompt = get_target_prompt()
        pure_system_prompt = get_pure_target_prompt()

        original_demos = ""
        optimized_demos = ""
        original_demos += f"question: {question}\n requirements: {requirements}\n"
        optimized_demos += f"question: {question}\n requirements: {requirements}\n"
        print(i)
        i+=1
        for key, value in data.items():
            print(f"key = {key}, value = {value}")
            original_demos += str(key)
            original_demos += ": "
            original_demos += value["original"]
            original_demos += "\n"
            if key == target_demo:
                optimized_demos += str(key)
                optimized_demos += ": "
                optimized_demos += value["replaced"]
                optimized_demos += "\n"
            else:
                optimized_demos += str(key)
                optimized_demos += ": "
                optimized_demos += value["original"]
                optimized_demos += "\n"
        
        print(f"Original demos: {original_demos}")
        print(f"Optimized demos: {optimized_demos}")
        if compressed:
            le = 0 
            if len(optimized_demos) < 800:
                le = 25
            elif len(optimized_demos) < 1200:
                le = 50
            else:
                le = 100
            optimized_demos = compression_model.compress_prompt(
                optimized_demos,
                instruction="",
                question="",
                target_token=le,
            )
            original_demos = compression_model.compress_prompt(
                original_demos,
                instruction="",
                question="",
                target_token=le,
            )
            original_demos = original_demos["compressed_prompt"]
            optimized_demos = optimized_demos["compressed_prompt"]

        message_dict = {}
        message_dict["original_message"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": original_demos},
        ]
        message_dict["optimized_message"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": optimized_demos},
        ]
        # message_list.append(optimized_message)
        message_dict["pure_original_message"] = [
            {"role": "system", "content": pure_system_prompt},
            {"role": "user", "content": original_demos},
        ]
        # message_list.append(pure_original_message)
        message_dict["pure_optimized_message"] = [
            {"role": "system", "content": pure_system_prompt},
            {"role": "user", "content": original_demos},
        ]
        # message_list.append(pure_optimized_message)

        output_list.append(message_dict)
        # break
    
    return output_list

# def llama2_inference(dataset, question_dataset, compression_model, flag="increase", output_path="", compressed=True, common=False):


def qwen3_inference(dataset, question_dataset, compression_model, flag="increase", output_path="", compressed=True, common=False):
    """"""
    model_name =  "models/Qwen3-32B"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # selection_function = None
    # if flag=="decrease":
    #     selection_function = max
    # else:
    #     selection_function = min
    output_list = []
    message_list = dataset_process(
        dataset=dataset,
        question_dataset=question_dataset,
        flag=flag,
        compressed=compressed,
        compression_model=compression_model,
        common=common,
    )
    
    # if compressed is not None:
    #     message_dict = dataset
    for message_dict in tqdm(message_list):
        output_dict = {}
        for key, prompt in message_dict.items():
            messages = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            model_inputs = tokenizer(
                messages,
                return_tensors='pt',
                padding=True,
            ).to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=32768
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            del generated_ids
            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            output_dict[key] = content
            print(f"key = {key}, content = {content}")
            
        # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        # output_dict = {}
        # for i, single_generated_ids in enumerate(generated_ids):
        #     # Get the input length for the *current* item in the batch
        #     input_len = len(model_inputs.input_ids[i])
            
        #     # Parse the output for the current item
        #     output_ids = single_generated_ids[input_len:].tolist()
            
        #     try:
        #         # rindex finding 151668 (</think>)
        #         index = len(output_ids) - output_ids[::-1].index(151668)
        #     except ValueError:
        #         index = 0

        #     thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        #     content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        #     output_dict["i"] = content
        
        output_list.append(output_dict)

    if compressed:
        output_path = f"{output_path}/{flag}_compressed_qwen3.json"
    else:
        output_path = f"{output_path}/{flag}_without_compressed_qwen3.json"
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output_list, file, indent=4)
    
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def llama3_inference(dataset, question_dataset, compression_model, flag="increase", output_path="", compressed=True, common=False):
    """
    Llama3 as the large model to inference.
    """

    model_name = "models/Llama-3-8B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    message_list = dataset_process(
        dataset=dataset,
        question_dataset=question_dataset,
        flag=flag,
        compressed=compressed,
        compression_model=compression_model,
        common=common,
    )

    output_list = []
    for data_entry in tqdm(message_list):
        output_dict = {}
        for key, value in data_entry.items():
            outputs = pipeline(
                value,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            output = outputs[0]["generated_text"][-1]
            output_dict[key] = output
            del outputs
        output_list.append(output_dict)
    
    
    if compressed:
        output_path = f"{output_path}/{flag}_compressed_llama3.json"
    else:
        output_path = f"{output_path}/{flag}_without_compressed_llama3.json"
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output_list, file, indent=4)
    
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


def llama2_inference(dataset, question_dataset, compression_model, flag="increase", output_path="", compressed=True, common=False):
    """
    LLaMA-2 as the large model to inference.
    """

    model_name = "models/Llama-2-7b-chat-hf" 
    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.eos_token_id = tokenizer.pad_token_id

    message_list = dataset_process(
        dataset=dataset,
        question_dataset=question_dataset,
        flag=flag,
        compressed=compressed,
        compression_model=compression_model,
        common=common,
    )

    output_list = []
    for data_entry in tqdm(message_list):
        output_dict = {}
        for key, value in data_entry.items():
            if isinstance(value, list) and isinstance(value[0], dict) and "role" in value[0]:
                prompt = ""
                for msg in value:
                    role = msg["role"]
                    content = msg["content"]
                    prompt += f"<|{role}|>\n{content}\n"
            else:
                prompt = str(value)  # fallback
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(model.device)
            output_ids = model.generate(
                input_ids = input_ids,
                max_new_tokens = 100,
            )
            output_text = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
            output_dict[key] = output_text
            del output_ids
        output_list.append(output_dict)

    if compressed:
        output_path = f"{output_path}/{flag}_compressed_llama2.json"
    else:
        output_path = f"{output_path}/{flag}_without_compressed_llama2.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output_list, file, indent=4)

    # del pipeline
    gc.collect()
    torch.cuda.empty_cache()


def phi4_inference(dataset, question_dataset, compression_model, flag="increase", output_path="", compressed=True, common=False):
    """
    Phi-4 of Microsoft as the large model to inference
    """

    model_name = "models/phi-4"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": "auto"},
       device_map="auto",
    )
    
    message_list = dataset_process(
        dataset=dataset,
        question_dataset=question_dataset,
        flag=flag,
        compressed=compressed,
        compression_model=compression_model,
        common=common,
    )

    output_list = []
    for data_entry in tqdm(message_list):
        output_dict = {}
        for key, value in data_entry.items():
            outputs = pipeline(
                value, 
                max_new_tokens=128,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                )
            output = outputs[0]["generated_text"][-1]
            output_dict[key] = output
            del outputs
        
        output_list.append(output_dict)
    if compressed is not None:
        output_path = f"{output_path}/{flag}_compressed_phi4.json"
    else:
        output_path = f"{output_path}/{flag}_without_compressed_phi4.json"
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output_list, file, indent=4)


    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
        
def deepseekr1_inference():
    """
    
    """
    
def mistral2_inference(dataset, question_dataset, compression_model, flag="increase", output_path="", compressed=True, common=False):
    """"""
    # mistral_models_path = "MISTRAL_MODELS_PATH"
    
    tokenizer = MistralTokenizer.v1()
    
    # completion_request = ChatCompletionRequest(messages=[UserMessage(content="Explain Machine Learning to me in a nutshell.")])
    
    # tokens = tokenizer.encode_chat_completion(completion_request).tokens

    model_name = "models/Mistral-7B-Instruct-v0.2"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
    )
    model.to("cuda")

    message_list = dataset_process(
        dataset=dataset,
        question_dataset=question_dataset,
        compressed=compressed,
        compression_model=compression_model,
        flag=flag,
        common=common,
    )
    
    output_list = []
    for data_entry in tqdm(message_list):
        output_dict = {}
        for key, value in data_entry.items():
            prompt = value[0]["content"] + "\n" + value[1]["content"]
            
            message = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
            tokens = tokenizer.encode_chat_completion(message).tokens
            generated_ids = model.generate(tokens, max_new_tokens=1000, do_sample=True)
            result = tokenizer.decode(generated_ids[0].tolist())
            del generated_ids
            output_dict[key] = result
        
        output_list.append(output_dict)
    
    if compressed is not None:
        output_path = f"{output_path}/{flag}_compressed_mistral.json"
    else:
        output_path = f"{output_path}/{flag}_without_compressed_mistral.json"
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output_list, file, indent=4)

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    

