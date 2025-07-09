"""
LangChain ToolCallingAgent Demo
================================================================================

Module: script07.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates using ToolCallingAgent for automatic tool calling
    and execution with LangChain. Shows how to create an agent that can
    automatically select and execute tools based on user queries.
    
    Key features:
    - Uses ToolCallingAgent for automatic tool selection and execution
    - Demonstrates agent workflow with add and subtract functions as tools
    - Uses AzureChatOpenAI with streamlined agent execution
    
    Based on: https://python.langchain.com/docs/how_to/agent_executor/
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Initialize LLM using init_chat_model
llm = init_chat_model(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model_provider="azure_openai",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    temperature=0.1,
    max_tokens=1000
)

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The sum of a and b
    """
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first number.
    
    Args:
        a: First number
        b: Second number to subtract from a
        
    Returns:
        The result of a - b
    """
    return a - b

def demo_tool_calling_agent():
    """Demonstrate ToolCallingAgent with automatic tool execution"""
    print("\n" + "="*60)
    print("Demo: ToolCallingAgent with Automatic Tool Execution")
    print("="*60)
    
    # Create tools list
    tools = [add, subtract]
    
    # Create prompt template for the agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can perform mathematical calculations using the available tools."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create the tool calling agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Test queries
    queries = [
        "What is 45 + 28?",
        "What is 100 - 37?", 
        "Calculate 25 + 15, then subtract 10 from that result"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Question: {query}")
        print("-" * 40)
        
        try:
            result = agent_executor.invoke({"input": query})
            print(f"Agent result: {result['output']}")
        except Exception as e:
            print(f"Error: {e}")
        
        print()

def main():
    """Main function to run all demos"""
    print("LangChain ToolCallingAgent Demo")
    print("=" * 80)
    
    demo_tool_calling_agent()
    
    print("\n" + "="*80)
    print("Demo completed!")

if __name__ == "__main__":
    main() 