"""
LangChain Agent Tool Calling Demo
================================================================================

Module: script02.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates LangChain agent tool calling functionality.
    Uses create_tool_calling_agent and AgentExecutor with a mock weather tool.
"""

import os
import random
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from typing import List

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

def get_weather(location: str) -> str:
    """
    Mock function to get weather information for a given location.
    
    Args:
        location (str): The location to get weather for
        
    Returns:
        str: Weather information in a formatted string
    """
    # Mock weather conditions
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "snowy", "windy"]
    temperatures = list(range(-5, 35))  # Temperature range from -5 to 35°C
    
    # Generate random weather data
    condition = random.choice(conditions)
    temperature = random.choice(temperatures)
    humidity = random.randint(30, 90)
    wind_speed = random.randint(5, 25)
    
    weather_info = f"""Weather in {location}:
    - Condition: {condition.title()}
    - Temperature: {temperature}°C
    - Humidity: {humidity}%
    - Wind Speed: {wind_speed} km/h"""
    
    return weather_info

def create_weather_tool():
    """Create the weather tool for the agent"""
    return Tool(
        name="get_weather",
        description="Get current weather information for a specific location. Use this when users ask about weather conditions.",
        func=get_weather
    )

def demo_agent_tool_calling():
    """Demonstrate agent with tool calling functionality"""
    print("\n" + "="*60)
    print("Demo: Agent Tool Calling with Weather Tool")
    print("="*60)
    
    # Create tools
    tools = [create_weather_tool()]
    
    # Create prompt template for agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that can provide weather information.
        When users ask about weather, use the get_weather tool to get current conditions.
        Always be friendly and provide helpful responses."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Test queries
    queries = [
        "What's the weather like in Hanoi?",
        "Can you tell me the weather conditions in Ho Chi Minh City?",
        "I'm planning a trip to Da Nang. How's the weather there?",
        "What's the current weather in Paris?",
        "Hello, how are you?",  # Non-weather query to test general conversation
    ]
    
    for query in queries:
        print(f"\nUser Query: {query}")
        print("-" * 50)
        
        try:
            response = agent_executor.invoke({"input": query})
            print(f"Agent Response: {response['output']}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "="*50)

def main():
    """Main function to run the agent demo"""
    print("LangChain Agent Tool Calling Demo")
    print("=" * 80)
    
    # Run the demo
    demo_agent_tool_calling()

if __name__ == "__main__":
    main() 