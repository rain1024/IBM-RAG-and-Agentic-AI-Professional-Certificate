"""
LangChain Function Calling Demo
================================================================================

Module: script03.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates LangChain function calling functionality with LLM.
    Uses AzureChatOpenAI with add and subtract functions as tools.
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

def demo_basic_function_calling():
    """Demonstrate basic function calling with LLM"""
    print("\n" + "="*60)
    print("Demo: Basic Function Calling")
    print("="*60)
    
    # Create tools list
    tools = [add, subtract]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Test queries
    queries = [
        "What is 25 + 37?",
        "Calculate 100 - 42",
        "What's 15.5 plus 8.3?",
        "Subtract 7.2 from 20.8",
        "What is 3 + 12? Also, what is 11 + 49?",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        # Invoke LLM with tools
        result = llm_with_tools.invoke([HumanMessage(content=query)])
        
        # Print tool calls if any
        if result.tool_calls:
            print("Tool calls:")
            for tool_call in result.tool_calls:
                print(f"  - {tool_call['name']}: {tool_call['args']}")
        else:
            print("No tool calls detected")
            print(f"Response: {result.content}")

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

def demo_streaming_function_calls():
    """Demonstrate streaming with function calls"""
    print("\n" + "="*60)
    print("Demo: Streaming Function Calls")
    print("="*60)
    
    # Create tools list
    tools = [add, subtract]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    query = "Calculate 123 + 456 and then subtract 100 from the result"
    print(f"Query: {query}")
    print("-" * 40)
    
    print("Streaming response:")
    
    # Stream the response
    gathered = None
    for chunk in llm_with_tools.stream([HumanMessage(content=query)]):
        if gathered is None:
            gathered = chunk
        else:
            gathered = gathered + chunk
        
        # Show tool calls as they build up
        if hasattr(gathered, 'tool_calls') and gathered.tool_calls:
            print(f"Current tool calls: {len(gathered.tool_calls)}")
            for i, tool_call in enumerate(gathered.tool_calls):
                print(f"  {i+1}. {tool_call.get('name', 'Unknown')}: {tool_call.get('args', {})}")
    
    print(f"\nFinal gathered tool calls: {gathered.tool_calls if hasattr(gathered, 'tool_calls') else 'None'}")

def demo_few_shot_prompting():
    """Demonstrate few-shot prompting for better tool usage"""
    print("\n" + "="*60)
    print("Demo: Few-Shot Prompting for Complex Operations")
    print("="*60)
    
    from langchain_core.messages import AIMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    
    # Create tools list
    tools = [add, subtract]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Create few-shot examples
    examples = [
        HumanMessage(
            "What's 100 plus 50 minus 25?", 
            name="example_user"
        ),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {"name": "add", "args": {"a": 100, "b": 50}, "id": "1"}
            ],
        ),
        ToolMessage("150", tool_call_id="1"),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {"name": "subtract", "args": {"a": 150, "b": 25}, "id": "2"}
            ],
        ),
        ToolMessage("125", tool_call_id="2"),
        AIMessage(
            "The result of 100 plus 50 minus 25 is 125.",
            name="example_assistant",
        ),
    ]
    
    # Create prompt with examples
    system = """You are a helpful calculator assistant. Always use tools for mathematical operations.
    Follow the order of operations and use the provided examples as guidance."""
    
    few_shot_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        *examples,
        ("human", "{query}"),
    ])
    
    # Create chain
    chain = {"query": RunnablePassthrough()} | few_shot_prompt | llm_with_tools
    
    # Test complex query
    query = "Calculate 200 plus 75 minus 50"
    print(f"Query: {query}")
    print("-" * 40)
    
    result = chain.invoke(query)
    
    if result.tool_calls:
        print("Tool calls made:")
        for tool_call in result.tool_calls:
            print(f"  - {tool_call['name']}: {tool_call['args']}")
    else:
        print("No tool calls detected")
        print(f"Response: {result.content}")

def main():
    """Main function to run all demos"""
    print("LangChain Function Calling Demo")
    print("=" * 80)
    
    # Run all demos
    demo_basic_function_calling()
    demo_tool_execution()
    demo_streaming_function_calls()
    demo_few_shot_prompting()
    
    print("\n" + "="*80)
    print("Demo completed!")

if __name__ == "__main__":
    main() 