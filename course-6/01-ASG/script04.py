"""
Simplest Structured Tool Demo in LangChain
================================================================================

Module: script04.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    The simplest example of structured tools in LangChain.
    Uses Pydantic models to define tool input schema for better type safety.
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

# Define input schema for structured tool
class CalculatorInput(BaseModel):
    """Input schema for calculator tool"""
    a: float = Field(description="First number")
    b: float = Field(description="Second number")
    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
        description="Mathematical operation to perform"
    )

class GreetingInput(BaseModel):
    """Input schema for greeting tool"""
    name: str = Field(description="Name of the person to greet")
    language: Literal["english", "vietnamese", "spanish"] = Field(
        default="english",
        description="Language for greeting"
    )

# Define structured tools
@tool(args_schema=CalculatorInput)
def structured_calculator(a: float, b: float, operation: str) -> str:
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
        
        return f"Result: {a} {operation} {b} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"

@tool(args_schema=GreetingInput)
def structured_greeting(name: str, language: str = "english") -> str:
    """Generate personalized greetings with structured input validation."""
    greetings = {
        "english": f"Hello, {name}! Nice to meet you!",
        "vietnamese": f"Xin chào, {name}! Rất vui được gặp bạn!",
        "spanish": f"¡Hola, {name}! ¡Encantado de conocerte!"
    }
    
    return greetings.get(language, greetings["english"])

def create_structured_tool_agent():
    """Create agent with structured tools"""
    
    # List of structured tools
    tools = [structured_calculator, structured_greeting]
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with access to structured tools:
        - structured_calculator: performs math operations (add, subtract, multiply, divide)
        - structured_greeting: generates personalized greetings in different languages
        
        Use these tools when appropriate. The tools have structured input validation."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

def run_structured_tool_demo():
    """Run structured tool demonstration"""
    print("\n" + "="*60)
    print("SIMPLEST STRUCTURED TOOL DEMO IN LANGCHAIN")
    print("="*60)
    
    # Create agent
    agent = create_structured_tool_agent()
    
    # Demo queries
    demo_queries = [
        "Calculate 15 + 25",
        "Multiply 7 by 8",
        "Divide 100 by 4",
        "Greet John in English",
        "Say hello to Maria in Spanish",
        "Greet Linh in Vietnamese",
        "What's 50 minus 30?"
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
    
    print("\nStructured Tool Demo completed!")

if __name__ == "__main__":
    run_structured_tool_demo() 