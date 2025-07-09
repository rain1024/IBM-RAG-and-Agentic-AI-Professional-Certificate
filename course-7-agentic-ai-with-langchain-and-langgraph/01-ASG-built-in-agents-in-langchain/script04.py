"""
Simple Pandas Agent Demonstration
================================================================================

Module: script04.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates the simplest usage of LangChain's pandas agent
    for executing commands on pandas dataframes using natural language.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.7,
    max_tokens=500
)

def main():
    """Simple pandas agent demonstration"""
    print("Simple Pandas Agent Demo")
    print("=" * 40)
    
    # Create simple sample data
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000, 60000, 70000, 55000, 65000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT']
    }
    
    df = pd.DataFrame(data)
    print("Sample DataFrame:")
    print(df)
    print()
    
    # Create pandas agent
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="tool-calling",
        verbose=True,
        allow_dangerous_code=True
    )
    
    # Execute simple commands
    commands = [
        "What is the shape of this dataframe?",
        "What is the average salary?",
        "Who has the highest salary?",
        "How many people work in IT department?"
        "Thêm một dòng vào dataframe Jame 20 tuổi có lương 1000000 thuộc phòng IT"
    ]
    
    print("Executing commands with pandas agent...")
    print()
    
    for i, command in enumerate(commands, 1):
        print(f"{i}. Command: {command}")
        try:
            result = agent.invoke(command)
            print(f"   Answer: {result['output']}")
        except Exception as e:
            print(f"   Error: {e}")
        print("-" * 40)

if __name__ == "__main__":
    main() 