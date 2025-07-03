#!/usr/bin/env python3
"""
LangChain Demo with Azure OpenAI
A simple demonstration of using LangChain with Azure OpenAI for prompt engineering.
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

def setup_azure_openai():
    """
    Set up Azure OpenAI client using environment variables.
    
    Required environment variables:
    - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
    - AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint
    - AZURE_OPENAI_API_VERSION: API version (e.g., "2023-12-01-preview")
    - AZURE_OPENAI_DEPLOYMENT_NAME: Your deployment name
    """
    
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=0.7,
            max_tokens=1000
        )
        return llm
    except Exception as e:
        print(f"Error setting up Azure OpenAI: {e}")
        print("Please ensure all required environment variables are set.")
        return None

def demo_prompt_components(llm):
    demo_description = """
    Prompt components are the building blocks of a prompt.
    They are used to create a prompt that is used to generate a response.
    
    There are four types of prompt components:
    - Context
    - Instructions
    - Input data
    - Output indicator

    Context: The context is the background information that the model uses to generate a response.
    Instructions: The instructions are the instructions that the model follows to generate a response.
    Input data: The input data is the input that the model uses to generate a response.
    Output indicator: The output indicator is the indicator that the model uses to generate a response.
    """
    print(f"Demo Description: {demo_description}")

    system_message = """
    You are an expert sentiment analyzer
    Classify given text as positive, negative or neutral
    """

    human_message = """
    Text: "I am very happy with the service"
    Sentiment:
    """

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message)
    ]
    
    response = llm.invoke(messages)
    print(f"{response.content}")

def main():
    llm = setup_azure_openai()
    demo_prompt_components(llm)

if __name__ == "__main__":
    main()
