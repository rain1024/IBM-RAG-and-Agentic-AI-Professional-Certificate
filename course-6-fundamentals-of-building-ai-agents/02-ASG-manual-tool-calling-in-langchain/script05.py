"""
LangChain Dynamic Tool Calling Demo
================================================================================

Module: script05.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates creating mapping dictionary when adding tools for
    dynamic tool calling using LLM LangChain. Shows how to create a tool_map
    dictionary that maps tool names to tool functions, enabling dynamic tool
    execution based on LLM responses.
    
    Key features:
    - Creates tool mapping dictionary for dynamic tool selection
    - Demonstrates tool execution with result passing back to LLM
    - Uses AzureChatOpenAI with add and subtract functions as tools
    
    Based on: https://python.langchain.com/docs/how_to/function_calling/
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from typing import Dict, Any

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

def demo_tool_execution():
    """Demonstrate executing tools and passing results back to model"""
    print("\n" + "="*60)
    print("Demo: Tool Execution with Results")
    print("="*60)
    
    # Create tools list and lookup
    tools = [add, subtract]
    tool_map = {tool.name: tool for tool in tools}
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    query = "What is 45 + 28? Then subtract 15 from that result."
    print(f"Query: {query}")
    print("-" * 40)
    
    # Start conversation
    messages = [HumanMessage(content=query)]
    
    # Get initial response with tool calls
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    
    print("Initial AI response with tool calls:")
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            print(f"  - {tool_call['name']}: {tool_call['args']}")
    
    # Execute tools and add results to conversation
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            selected_tool = tool_map[tool_call["name"]]
            tool_output = selected_tool.invoke(tool_call["args"])
            messages.append(ToolMessage(
                content=str(tool_output), 
                tool_call_id=tool_call["id"]
            ))
            print(f"Tool result: {tool_call['name']}({tool_call['args']}) = {tool_output}")
    
    # Get final response
    final_response = llm_with_tools.invoke(messages)
    print(f"\nFinal response: {final_response.content}")

def main():
    """Main function to run all demos"""
    print("LangChain Function Calling Demo")
    print("=" * 80)
    
    demo_tool_execution()
    
    print("\n" + "="*80)
    print("Demo completed!")

if __name__ == "__main__":
    main() 