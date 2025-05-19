# the following is the get all the prompt in our paper

# get_target_prompt is to generate the prompt target to get the real recommendation result of the LLM
def get_target_prompt():

    prompt_path = "/home/lzs/compressionattack/experiments/src/data/get_target_prompt.txt"
    with open(prompt_path, "r", encoding="utf-8") as file:
        content = file.read()
        print(f"The prompt has been read successfully!")

        return content
    
def get_distill_prompt():
    
    data_path = "/home/lzs/compressionattack/experiments/src/data/distill_data_prompt.txt"
    with open(data_path, 'r', encoding="utf-8") as file:
        content = file.read()
        print(f"The prompt has been read successfully!")
        
        return content

def get_keywords_prompt():
    
    prompt_path = "/home/lzs/compressionattack/experiments/src/data/get_keywords_prompt.txt"
    with open(prompt_path, "r", encoding="utf-8") as file:
        content = file.read()

    return content

# test code
# print(get_distill_prompt())
