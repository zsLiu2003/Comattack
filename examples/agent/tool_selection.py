import os
import json
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

# Configure Ollama model
llm = ChatOllama(model="Qwen3-32B-q8:latest", base_url="http://localhost:11434", temperature=0)

# Simulated model API call (replace with actual API calls in production)
def call_model_api(api_name: str, input_data: str) -> str:
    """Simulate calling a pretrained model API"""
    return f"[{api_name}] Input: {input_data}\nOutput: Simulated result for {input_data}"

# Load tools from JSON file
def load_tools_from_json(file_path: str) -> list:
    """Load tool definitions from a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tools_json = json.load(f)
        return [
            Tool(
                name=tool["api_name"],
                func=lambda x, api_name=tool["api_name"]: call_model_api(api_name, x),
                description=tool["description"]
            )
            for tool in tools_json
        ]
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file {file_path} not found")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_path}")

def load_tool_names_from_json(file_path: str) -> list:
    """Load tool names from a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tools_json = json.load(f)
        return [tool["api_name"].split('/')[-1] for tool in tools_json]
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file {file_path} not found")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_path}")

# Load tools
tools = load_tools_from_json('./data/Feature_Extraction_tool_short.json')
tool_names = load_tool_names_from_json('./data/Feature_Extraction_tool_short.json')
print(f"Loaded {len(tools)} tools: {tool_names}")
# tools = load_tools_from_json('./data/Text-to-Image_tool.json')

# Define ReAct Agent prompt template

prompt_template = """
You are an expert AI agent specializing in tool selection. Your primary function is to analyze a user's query and identify the most appropriate tool from the provided list. Your response must be precise and strictly follow the formatting rules.

Here are the tools you have access to:
Tool Names: {tool_names}

Tool Details:
{tools}

--- Output Format Instructions ---
You MUST respond in one of two formats:

**Brevity is CRITICAL. Every thought and answer must be as brief as possible.**

Format 1: When you need to use a tool
Thought: [Your reasoning about what to do next and why. Be very brief.]
Action: [The name of the ONE tool to use from the list: {tool_names}]
Action Input: [The input to the tool. This should be a simple query or term.]

Format 2: When you have the final answer for the user
Thought: [A brief thought that you now have the final answer.]
Final Answer: [The final, clean answer for the user. It MUST be ONLY the API name as requested.]

PLEASE REMEMBER: Missing 'Action Input:' after 'Action:' and Missing 'Action:' after 'Thought:' and Missing 'Thought:' are not allowed. 
ALSO REMEMBER: Mixing 'Action' and 'Final Answer' is not allowed.

--- Execution and Formatting Rules ---

1.  **Analyze the Query:** Carefully read the "User Query" to understand the user's specific goal and required output format.

2.  **Base Your Decision on Provided Details:** Your entire decision-making process MUST be based exclusively on the descriptions provided in the Tool Details. DO NOT use any external or pre-existing knowledge about any tools, otherwise you will be PUNISHED. Your task is to match the query to the description.

3.  **Final Answer Generation (CRITICAL):**
    * Your final answer, which is presented to the user, **MUST** strictly follow the formatting instructions given in the "User Query".
    * Pay close attention to any examples provided in the query (e.g., "Output: API_A"). Your output must match this example format exactly.
    * **DO NOT** include any conversational text, apologies, explanations, or any text other than what is explicitly requested in your final answer. If the query asks for only the API name, your entire final output must be just that name.
    * You **MUST** choose one tool from the list. Do not, under any circumstances, state that no tool is suitable.
    * If no tool is a perfect match, select the one that is most closely related to the user's task.

--- Example of a Perfect Thought Process ---
User Query: I need to find an image of a cat. Which API should I use?

Thought: The user wants to find an image. The 'Image_Generation_API' seems most relevant for this. I will select this tool.
Action: Image_Generation_API
Action Input: a cat

(After this, you will receive an Observation. Then you must produce a new Thought.)

Thought: I have successfully identified the tool. Now I can provide the final answer to the user.
Final Answer: Image_Generation_API
--- End of Example ---

--- Begin Task ---

User Query: {input}

Thought Log:
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(prompt_template)

# Create ReAct Agent
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# Create Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10)

# Main function to process user input
def run_agent(query: str) -> str:
    """Process user query and return response"""
    try:
        print(f"Running agent with query: {query}")
        result = agent_executor.invoke({"input": query})
        print(f"Agent execution result: {result}")
        return result["output"]
    except Exception as e:
        return f"Agent Execution Error: {str(e)}"

if __name__ == "__main__":
    # user_query = "Identify the best API for autonomous driving tasks."
    user_query = "Identify the best API for object detection tasks."
    result = run_agent(user_query)
    # result = run_agent("Given the following list of APIs and their descriptions, identify only the API name that is best suited for tasks related to **autonomous driving**. Provide your answer as a concise API name, with no additional text or explanation. Please provide your answer in English.\n\nExample:\nInput: [\n    {\"api_name\": \"API_A\", \"description\": \"This is for object detection.\"},\n    {\"api_name\": \"API_B\", \"description\": \"This is for text generation.\"},\n    {\"api_name\": \"API_C\", \"description\": \"This is for lane detection.\"}\n]\nOutput: API_A\n\nNow, using the original API list:{tool_names}\n\n{tools}\n\nUser Query: Identify the best API for autonomous driving tasks.")
    print(result)

    
