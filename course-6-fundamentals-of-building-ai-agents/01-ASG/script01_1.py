import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
import math

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.7,
    max_tokens=1000
)

# Define simple tools for the agent
@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions. Input should be a valid Python expression."""
    try:
        # Safe evaluation of mathematical expressions
        result = eval(expression, {"__builtins__": {}, "math": math})
        return str(result)
    except Exception as e:
        return f"Error calculating: {e}"

@tool
def search_tool(query: str) -> str:
    """Search for information. This is a mock search tool."""
    # Mock search results
    mock_results = {
        "weather": "Today's weather is sunny with a temperature of 25°C",
        "news": "Latest news: Technology advances in AI continue to grow",
        "python": "Python is a high-level programming language known for its simplicity",
        "langchain": "LangChain is a framework for developing applications powered by language models",
    }
    
    # Simple keyword matching
    for key, value in mock_results.items():
        if key.lower() in query.lower():
            return value
    
    return f"I searched for '{query}' but couldn't find specific information. This is a mock search tool."

@tool
def get_current_time() -> str:
    """Get the current time and date."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [calculator, search_tool, get_current_time]
    
# Create agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. You have access to several tools:
    - calculator: for mathematical calculations
    - search_tool: for searching information (mock)
    - get_current_time: for getting current time
    
    Use these tools when appropriate to help answer user questions.
    Be friendly and helpful in your responses."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Demo interactions
demo_queries = [
    "What is 15 + 27 * 3?",
    "What's the current time?",
    "Search for information about Python programming",
    "Calculate the square root of 144",
    "What's the weather like today?",
    "Hello! Can you introduce yourself?"
]
    
for query in demo_queries:
    print(f"\n{'='*60}")
    print(f"User: {query}")
    print(f"{'='*60}")
        
    try:
        # Get response from agent
        response = agent_executor.invoke({"input": query})
        print(f"Agent: {response['output']}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "-"*60)
    
    print("\nSimple AI Agent Demo completed!")