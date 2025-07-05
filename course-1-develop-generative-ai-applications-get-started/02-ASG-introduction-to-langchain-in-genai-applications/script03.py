#!/usr/bin/env python3
"""
LangChain Agent with Tools Demo
A comprehensive demonstration of creating and using agents with tools in LangChain.
Based on: https://python.langchain.com/docs/tutorials/agents/
"""

import os
import math
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()

print("=" * 60)
print("LangChain Agent with Tools Demo")
print("=" * 60)

# Create custom tools
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate_circle_area(radius: float) -> float:
    """Calculate the area of a circle given its radius."""
    return math.pi * radius * radius

# Set up the language model
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.1,  # Lower temperature for more consistent tool usage
    max_tokens=1000
)

# Create tools list
tools = [multiply, add, get_current_time]


print(f"✓ Total tools available: {len(tools)}")
for tool in tools:
    print(f"  - {tool.name}: {tool.description}")

# Create the agent without memory first
print("\n" + "=" * 60)
print("Creating Agent (Stateless)")
print("=" * 60)

agent_executor = create_react_agent(llm, tools)

def run_agent_example(query: str, config: Optional[dict] = None):
    """Helper function to run agent and display results."""
    print(f"\n🤖 Query: {query}")
    print("-" * 40)
    
    try:
        messages = [{"role": "user", "content": query}]
        
        if config:
            # Stream with config (for memory)
            for step in agent_executor.stream(
                {"messages": messages}, 
                config, 
                stream_mode="values"
            ):
                if step["messages"]:
                    last_message = step["messages"][-1]
                    if hasattr(last_message, 'content') and last_message.content:
                        print(f"📝 {last_message.content}")
        else:
            # Stream without config (stateless)
            for step in agent_executor.stream(
                {"messages": messages}, 
                stream_mode="values"
            ):
                if step["messages"]:
                    last_message = step["messages"][-1]
                    if hasattr(last_message, 'content') and last_message.content:
                        print(f"📝 {last_message.content}")
                        
    except Exception as e:
        print(f"❌ Error: {e}")

# Example 1: Basic math operations
print("\n📊 Example 1: Basic Math Operations")
run_agent_example("What is 25 multiplied by 4, then add 10 to the result?")

# Example 2: Current time
print("\n📊 Example 2: Getting Current Time")
run_agent_example("What time is it now?")

# Example 3: Circle area calculation
print("\n📊 Example 3: Circle Area Calculation")
run_agent_example("What is the area of a circle with radius 5?")

# Example 4: Complex calculation
print("\n📊 Example 4: Complex Calculation")
run_agent_example("Calculate the area of a circle with radius 7, then multiply the result by 2")