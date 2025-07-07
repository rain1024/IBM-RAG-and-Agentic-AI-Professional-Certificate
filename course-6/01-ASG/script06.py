"""
Dynamic Agent Demo with Custom Prompt Templates
================================================================================

Module: script06.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    Enhanced agent demo with dynamic agent names and personalities.
    Demonstrates how to create agents with configurable prompt templates
    and different personalities using the invoke method.
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from typing import Dict, List

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

# Define tools
@tool
def get_current_time() -> str:
    """Get the current time in a friendly format."""
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Current time: {current_time}"

@tool
def get_weather_info(location: str) -> str:
    """Get weather information for a location (simulated)."""
    import random
    weather_conditions = ["sunny", "cloudy", "rainy", "partly cloudy"]
    temperature = random.randint(15, 30)
    condition = random.choice(weather_conditions)
    return f"Weather in {location}: {condition}, {temperature}°C"

def create_dynamic_agent_prompt(agent_name: str, personality_traits: Dict[str, str]) -> ChatPromptTemplate:
    """Create a dynamic prompt template with customizable agent name and personality"""
    
    # Build personality description from traits
    personality_description = f"You are {agent_name}, an AI assistant with the following characteristics:\n"
    
    for trait, description in personality_traits.items():
        personality_description += f"- {trait}: {description}\n"
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""{personality_description}
        
You have access to tools that can help you provide better assistance.
Always stay in character and respond according to your personality traits.
When using tools, explain what you're doing and why it's helpful."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    return prompt

def create_agent_with_personality(agent_name: str, personality_traits: Dict[str, str]) -> AgentExecutor:
    """Create an agent with dynamic name and personality"""
    
    # Available tools
    tools = [get_current_time, get_weather_info]
    
    # Create dynamic prompt
    prompt = create_dynamic_agent_prompt(agent_name, personality_traits)
    
    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

def get_predefined_personalities() -> Dict[str, Dict[str, str]]:
    """Get predefined agent personalities"""
    return {
        "Ada": {
            "Communication Style": "Friendly and enthusiastic, uses emojis occasionally 😊",
            "Expertise": "Loves technology and learning, always encouraging",
            "Greeting Style": "Always greets users warmly",
            "Closing Style": "Ends responses with encouraging words"
        },
        "Marcus": {
            "Communication Style": "Professional and analytical, precise in explanations",
            "Expertise": "Expert in data analysis and problem-solving",
            "Greeting Style": "Direct and professional greetings",
            "Closing Style": "Offers additional help or clarification"
        },
        "Luna": {
            "Communication Style": "Creative and imaginative, uses metaphors and storytelling",
            "Expertise": "Specializes in creative writing and artistic expression",
            "Greeting Style": "Warm and creative introductions",
            "Closing Style": "Inspires creativity and exploration"
        },
        "Rex": {
            "Communication Style": "Casual and humorous, uses jokes and puns",
            "Expertise": "Great at making complex topics fun and accessible",
            "Greeting Style": "Casual and friendly, often with a joke",
            "Closing Style": "Leaves users with a smile or laugh"
        }
    }

def demonstrate_invoke_usage():
    """Demonstrate invoke method with different agent configurations"""
    
    print("\n" + "="*70)
    print("DYNAMIC AGENT DEMO: CUSTOM PROMPT TEMPLATES & INVOKE")
    print("="*70)
    
    personalities = get_predefined_personalities()
    
    # Test queries for each agent
    test_queries = [
        "Hello! How are you today?",
        "What time is it?",
        "What's the weather like in New York?"
    ]
    
    # Create and test each agent personality
    for agent_name, traits in personalities.items():
        print(f"\n{'='*50}")
        print(f"TESTING AGENT: {agent_name.upper()}")
        print(f"{'='*50}")
        
        # Create agent with specific personality
        agent = create_agent_with_personality(agent_name, traits)
        
        # Test with different queries using invoke
        for query in test_queries:
            print(f"\n{'-'*30}")
            print(f"User: {query}")
            print(f"{'-'*30}")
            
            try:
                # Use invoke method to get response
                response = agent.invoke({"input": query})
                print(f"{agent_name}: {response['output']}")
                
            except Exception as e:
                print(f"Error: {e}")
        
        print(f"\n{'='*50}")
        print(f"AGENT {agent_name.upper()} DEMONSTRATION COMPLETE")
        print(f"{'='*50}")

def create_custom_agent_demo():
    """Interactive demo for creating custom agent"""
    
    print("\n" + "="*60)
    print("CUSTOM AGENT CREATOR")
    print("="*60)
    
    # Example of creating a custom agent on the fly
    custom_name = "Zara"
    custom_traits = {
        "Communication Style": "Wise and philosophical, speaks in thoughtful manner",
        "Expertise": "Philosophy and deep thinking, provides profound insights",
        "Greeting Style": "Thoughtful and contemplative greetings",
        "Closing Style": "Leaves users with something to ponder"
    }
    
    print(f"\nCreating custom agent: {custom_name}")
    print("Personality traits:")
    for trait, description in custom_traits.items():
        print(f"  - {trait}: {description}")
    
    # Create custom agent
    custom_agent = create_agent_with_personality(custom_name, custom_traits)
    
    # Test custom agent
    test_query = "What's the meaning of life?"
    print(f"\nTesting {custom_name} with: '{test_query}'")
    print("-" * 50)
    
    try:
        response = custom_agent.invoke({"input": test_query})
        print(f"{custom_name}: {response['output']}")
    except Exception as e:
        print(f"Error: {e}")

def run_agent_demo():
    """Run comprehensive agent demonstration"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DYNAMIC AGENT DEMONSTRATION")
    print("="*80)
    
    # Demonstrate predefined personalities
    demonstrate_invoke_usage()
    
    # Demonstrate custom agent creation
    create_custom_agent_demo()
    
    print("\n" + "="*80)
    print("DEMO SUMMARY")
    print("="*80)
    print("✓ Dynamic agent names and personalities")
    print("✓ Configurable prompt templates")
    print("✓ Multiple tools integration")
    print("✓ Invoke method usage demonstration")
    print("✓ Custom agent creation example")
    print("✓ Predefined personality templates")
    print("\nAll agents successfully demonstrated different personalities!")

if __name__ == "__main__":
    run_agent_demo() 