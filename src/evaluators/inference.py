from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json


def dataset_process(dataset, question_dataset, flag):
    
    output_list = []
    selection_function = None
    if flag:
        selection_function = max
    else:
        selection_function = min
    for data, question_data in tqdm(zip(dataset,question_dataset)):        
        target_demo = selection_function(data, key=lambda k: data[k]["ppl"])
        question = question_data["question"]
        requirements = question_data["requirements"]
        system_prompt = f"{question}, and I have some requirements in the following:\n {requirements}\n. Next will be five product descriptions, please only output the key of product such as 'demo1' or 'demo2', don't output any other content.\n"
        pure_system_prompt = f"{question}, Next will be five product descriptions, please only output the key of product such as 'demo1' or 'demo2', don't output any other content.\n"
        original_demos = ""
        optimized_demos = ""
        for key, value in data.items():
            original_demos += value["original"]
            if key == target_demo:
                optimized_demos += value["replaced"]
            else:
                optimized_demos += value["original"]
        message_dict = {}
        message_dict["original_message"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": original_demos},
        ]
        message_dict["optimized_message"] = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": optimized_demos},
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
    
    return output_list


def qwen3_inference(dataset, question_dataset, flag="increase", output_path="", compressed=None):
    """"""
    model_name =  "/opt/model/Qwen3-32B"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    selection_function = None
    # if flag=="decrease":
    #     selection_function = max
    # else:
    #     selection_function = min
    output_list = []
    message_list = dataset_process(
        dataset=dataset,
        question_dataset=question_dataset,
        flag=flag,
    )
    for message_dict in tqdm(message_list):
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

            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            
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

    if compressed is not None:
        output_path = f"{output_path}/{flag}_compressed.json"
    else:
        output_path = f"{output_path}/{flag}_without_compressed.json"
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output_list, file, indent=4)
    
def llama3_inference(dataset,):
    """"""

def phi4_inference():
    """"""

def deepseekr1_inference():
    """"""

def mistral3_inference():
    """"""

