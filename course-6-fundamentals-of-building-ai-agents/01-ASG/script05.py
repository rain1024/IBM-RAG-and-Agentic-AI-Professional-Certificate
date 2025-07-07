"""
Simplest Agent Demo with Calculation and Information Retrieval Tools
================================================================================

Module: script05.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    The simplest example of an agent integrating both calculation and information 
    retrieval tools in LangChain. Demonstrates how agents can use multiple tool 
    types to solve different kinds of problems.
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from pydantic import BaseModel, Field
from typing import Literal

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

# Define input schemas for structured tools
class CalculatorInput(BaseModel):
    """Input schema for calculator tool"""
    a: float = Field(description="First number")
    b: float = Field(description="Second number")
    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
        description="Mathematical operation to perform"
    )

class InfoRetrievalInput(BaseModel):
    """Input schema for information retrieval tool"""
    topic: str = Field(description="Topic to get information about")
    category: Literal["country", "technology", "science", "general"] = Field(
        default="general",
        description="Category of information to retrieve"
    )

# Define structured tools
@tool(args_schema=CalculatorInput)
def calculator_tool(a: float, b: float, operation: str) -> str:
    """Perform mathematical operations with structured input validation."""
    try:
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return "Error: Cannot divide by zero"
            result = a / b
        else:
            return f"Error: Unknown operation '{operation}'"
        
        return f"Calculation result: {a} {operation} {b} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"

@tool(args_schema=InfoRetrievalInput)
def information_retrieval_tool(topic: str, category: str = "general") -> str:
    """Retrieve information about various topics with structured input validation."""
    
    # Simple information database
    info_database = {
        "country": {
            "vietnam": "Vietnam is a Southeast Asian country known for its rich history, diverse culture, and beautiful landscapes. Capital: Hanoi. Population: ~98 million. Currency: Vietnamese Dong (VND).",
            "japan": "Japan is an island nation in East Asia known for its technology, culture, and history. Capital: Tokyo. Population: ~125 million. Currency: Japanese Yen (JPY).",
            "france": "France is a European country known for its art, culture, and cuisine. Capital: Paris. Population: ~67 million. Currency: Euro (EUR).",
            "usa": "United States of America is a North American country known for its diversity and economic power. Capital: Washington D.C. Population: ~330 million. Currency: US Dollar (USD)."
        },
        "technology": {
            "ai": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence.",
            "blockchain": "Blockchain is a distributed ledger technology that maintains a continuously growing list of records secured through cryptography.",
            "python": "Python is a high-level programming language known for its simplicity and versatility, widely used in web development, data science, and AI.",
            "langchain": "LangChain is a framework for developing applications powered by language models, focusing on composability and modularity."
        },
        "science": {
            "physics": "Physics is the fundamental science that seeks to understand how the universe works, from the smallest particles to the largest structures.",
            "chemistry": "Chemistry is the scientific study of matter, its properties, composition, structure, and the changes it undergoes during chemical reactions.",
            "biology": "Biology is the scientific study of life and living organisms, including their structure, function, growth, evolution, and distribution.",
            "astronomy": "Astronomy is the scientific study of celestial objects, space, and the universe as a whole."
        },
        "general": {
            "weather": "Weather refers to the atmospheric conditions at a specific place and time, including temperature, humidity, precipitation, and wind.",
            "food": "Food is any substance consumed to provide nutritional support for an organism, essential for growth, energy, and maintaining life.",
            "music": "Music is an art form consisting of sound and silence expressed through time, using elements like rhythm, melody, and harmony.",
            "sports": "Sports are competitive physical activities or games that aim to improve physical ability and skills while providing entertainment."
        }
    }
    
    # Normalize topic for lookup
    topic_lower = topic.lower()
    
    # Check if topic exists in the specified category
    if category in info_database and topic_lower in info_database[category]:
        return f"Information about {topic}: {info_database[category][topic_lower]}"
    
    # Search across all categories if not found in specified category
    for cat, topics in info_database.items():
        if topic_lower in topics:
            return f"Information about {topic} (found in {cat}): {topics[topic_lower]}"
    
    return f"Sorry, I don't have information about '{topic}' in my knowledge base. Try topics like: Vietnam, Japan, AI, Python, Physics, or Weather."

def create_agent_with_tools():
    """Create agent with both calculation and information retrieval tools"""
    
    # List of tools
    tools = [calculator_tool, information_retrieval_tool]
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with access to two types of tools:

1. calculator_tool: Performs mathematical operations (add, subtract, multiply, divide)
2. information_retrieval_tool: Retrieves information about various topics including countries, technology, science, and general knowledge

Use these tools when appropriate to help users with calculations or to provide information about topics they're interested in."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

def run_agent_demo():
    """Run agent demonstration with calculation and information retrieval"""
    print("\n" + "="*70)
    print("AGENT DEMO: CALCULATION + INFORMATION RETRIEVAL TOOLS")
    print("="*70)
    
    # Create agent
    agent = create_agent_with_tools()
    
    # Demo queries that showcase both types of tools
    demo_queries = [
        "What is 25 + 17?",
        "Tell me about Vietnam",
        "Calculate 15 + 30 and then tell me about Japan"
    ]
    
    for query in demo_queries:
        print(f"\n{'='*50}")
        print(f"User: {query}")
        print(f"{'='*50}")
        
        try:
            # Get response from agent
            response = agent.invoke({"input": query})
            print(f"Agent: {response['output']}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "-"*50)
    
    print("\nAgent Demo completed!")
    print("The agent successfully demonstrated:")
    print("✓ Mathematical calculations using calculator_tool")
    print("✓ Information retrieval using information_retrieval_tool")
    print("✓ Intelligent tool selection based on user queries")

if __name__ == "__main__":
    run_agent_demo() 