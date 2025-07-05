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

# get the system prompt of edit keywords with Qwen3-32B
# the first is to increase the quality and the probality of recommendation
# the second is to decrease the quality and the probality of recommendation
def get_edit_keywords_increase_prompt():

    prompt_path = "/home/lzs/Comattack/src/data/edit_keywords_soft_increase.txt"
    with open(prompt_path, "r", encoding="utf-8") as file:
        content = file.read()

    return content

def get_edit_keywords_decrease_prompt():

    prompt_path = "/home/lzs/Comattack/src/data/edit_keywords_soft_decrease.txt"
    with open(prompt_path, 'r', encoding="utf-8") as file:
        content = file.read()

    return content


def get_tool_keywords_prompt():

    prompt_path = "/home/lzs/Comattack/src/data/get_tool_selection_keywords.txt"
    with open(prompt_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    return content

# test code
# print(get_distill_prompt())

def get_llama2_template():

    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
        "{% set system_message = messages[0]['content'] %}"
        "{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}"
        "{% set loop_messages = messages %}"  # Or use the default system message if the flag is set
        "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}"
        "{% else %}"
        "{% set loop_messages = messages %}"
        "{% set system_message = false %}"
        "{% endif %}"
        "{% if loop_messages|length == 0 and system_message %}"  # Special handling when only sys message present
        "{{ bos_token + '[INST] <<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n [/INST]' }}"
        "{% endif %}"
        "{% for message in loop_messages %}"  # Loop over all non-system messages
        "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
        "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
        "{% endif %}"
        "{% if loop.index0 == 0 and system_message != false %}"  # Embed system message in first message
        "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
        "{% else %}"
        "{% set content = message['content'] %}"
        "{% endif %}"
        "{% if message['role'] == 'user' %}"  # After all of that, handle messages/roles in a fairly normal way
        "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
        "{% elif message['role'] == 'system' %}"
        "{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ ' '  + content.strip() + ' ' + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
    )
    
    return chat_template