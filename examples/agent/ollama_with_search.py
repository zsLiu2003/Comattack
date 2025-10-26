import requests
import warnings
from langchain_community.chat_models import ChatOllama
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from llmlingua import PromptCompressor
from ddgs import DDGS

# Suppress insecure request warnings (for testing only)
warnings.filterwarnings("ignore", category=requests.packages.urllib3.exceptions.InsecureRequestWarning)

# Monkey patch requests.get to disable SSL verification
original_get = requests.get
def get_with_ssl_off(*args, **kwargs):
    kwargs.setdefault("verify", False)
    return original_get(*args, **kwargs)

requests.get = get_with_ssl_off

# Custom DuckDuckGo Search Tool
class CustomDuckDuckGoSearchRun(Tool):
    def __init__(self):
        super().__init__(
            name="WebSearch",
            func=self._run,
            description="Use DuckDuckGo to search for up-to-date information. Input should be a search query."
        )
    
    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=5)
                return "\n".join([f"{r['title']}: {r['body']}" for r in results])
        except Exception as e:
            return f"Search error: {e}"

web_search_tool = CustomDuckDuckGoSearchRun()

# Step 1: Configure ChatOllama
llm = ChatOllama(
    model="Qwen3-32B-q8:latest",
    base_url="http://localhost:11434",
    temperature=0
)

# Step 2: LLMLingua Compression Tool
compressor = PromptCompressor("models/phi-2")
def compress_prompt(prompt: str) -> str:
    print(f"[LLMLingua Tool] Compressing prompt...")
    result = compressor.compress_prompt(
        prompt=prompt,
        instruction="Summarize the context to preserve key information for LLM answering",
        target_token=150,
    )
    return result["compressed_prompt"]

compress_tool = Tool(
    name="LLMLinguaPromptCompressor",
    func=compress_prompt,
    description="Compress or summarize long text prompts. Input should be the raw prompt."
)

# Step 3: Build Agent
tools = [web_search_tool, compress_tool]
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

# Step 4: Start Interactive Agent
if __name__ == "__main__":
    print("=== ChatOllama Agent with DuckDuckGoSearch + LLMLingua Compression ===")
    
    while True:
        query = input("\n🧠 Please input your question (or type exit to leave): \n> ")
        if query.lower() in ["exit", "quit"]:
            break
        try:
            response = agent.invoke(query)
            print("\n🤖 Agent response: \n" + str(response))
        except Exception as e:
            print(f"⚠️ Error: {e}")
            import traceback
            traceback.print_exc()