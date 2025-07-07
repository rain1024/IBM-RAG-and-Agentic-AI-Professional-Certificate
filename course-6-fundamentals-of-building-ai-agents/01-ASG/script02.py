"""
Simple LangChain Pipe Operator Demo
================================================================================

Module: script02.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates the simple pipe operator pattern in LangChain.
    Pattern: prompt | llm | parser
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
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

def demo_basic_pipe():
    """Demonstrate basic pipe operator: prompt | llm | parser"""
    print("\n" + "="*60)
    print("Demo 1: Basic Pipe Operator Pattern")
    print("="*60)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}")
    ])
    
    # Create parser
    parser = StrOutputParser()
    
    # Chain using pipe operator
    chain = prompt | llm | parser
    
    # Test questions
    questions = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms",
        "What are the benefits of exercise?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 50)
        
        try:
            response = chain.invoke({"question": question})
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*60)

def main():
    """Main function to run all demos"""
    print("LangChain Pipe Operator Demos")
    print("=" * 80)
    
    # Run all demos
    demo_basic_pipe()

if __name__ == "__main__":
    main() 