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

def demo_simple_prompt(llm):
    """Demo 1: Simple direct prompt"""
    print("\n=== Demo 1: Simple Direct Prompt ===")
    
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="Explain what machine learning is in simple terms.")
    ]
    
    response = llm.invoke(messages)
    print(f"Response: {response.content}")

def demo_prompt_template(llm):
    """Demo 2: Using Prompt Templates"""
    print("\n=== Demo 2: Using Prompt Templates ===")
    
    # Create a prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in {subject}. Provide clear and accurate information."),
        ("human", "Explain {topic} in a way that a {audience} can understand.")
    ])
    
    # Format the prompt
    formatted_prompt = prompt_template.format_messages(
        subject="artificial intelligence",
        topic="neural networks",
        audience="beginner"
    )
    
    response = llm.invoke(formatted_prompt)
    print(f"Response: {response.content}")

def demo_llm_chain(llm):
    """Demo 3: Using LLM Chains"""
    print("\n=== Demo 3: Using LLM Chains ===")
    
    # Create a simple prompt template
    prompt = PromptTemplate(
        input_variables=["product", "audience"],
        template="Write a marketing tagline for a {product} targeting {audience}. Make it catchy and memorable."
    )
    
    # Create an LLM chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run the chain
    result = chain.invoke({
        "product": "smart home security system",
        "audience": "tech-savvy homeowners"
    })
    
    print(f"Marketing tagline: {result['text']}")

def demo_conversation_chain(llm):
    """Demo 4: Multi-turn conversation"""
    print("\n=== Demo 4: Multi-turn Conversation ===")
    
    conversation = [
        SystemMessage(content="You are a friendly programming tutor. Help students learn Python."),
        HumanMessage(content="I'm new to Python. Can you explain what a variable is?")
    ]
    
    # First response
    response1 = llm.invoke(conversation)
    print(f"Tutor: {response1.content}")
    
    # Add the AI's response and continue conversation
    conversation.append(response1)
    conversation.append(HumanMessage(content="Can you show me an example of creating a variable?"))
    
    # Second response
    response2 = llm.invoke(conversation)
    print(f"Tutor: {response2.content}")

def demo_different_temperatures(llm):
    """Demo 5: Different temperature settings"""
    print("\n=== Demo 5: Temperature Comparison ===")
    
    prompt = "Write a creative story opening about a robot discovering emotions."
    
    temperatures = [0.1, 0.7, 1.0]
    
    for temp in temperatures:
        llm_temp = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=temp,
            max_tokens=200
        )
        
        messages = [HumanMessage(content=prompt)]
        response = llm_temp.invoke(messages)
        
        print(f"\nTemperature {temp}:")
        print(f"Response: {response.content}")

def main():
    """Main function to run all demos"""
    print("LangChain + Azure OpenAI Demo")
    print("=" * 50)
    
    # Check if we have the required environment variables
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        print("\nExample .env file:")
        print("AZURE_OPENAI_API_KEY=your_api_key_here")
        print("AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
        print("AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name")
        print("AZURE_OPENAI_API_VERSION=2023-12-01-preview")
        return
    
    # Setup Azure OpenAI
    llm = setup_azure_openai()
    
    if llm is None:
        print("Failed to setup Azure OpenAI. Please check your configuration.")
        return
    
    print("Azure OpenAI client setup successful!")
    
    try:
        # Run all demos
        demo_simple_prompt(llm)
        demo_prompt_template(llm)
        demo_llm_chain(llm)
        demo_conversation_chain(llm)
        demo_different_temperatures(llm)
        
        print("\n" + "=" * 50)
        print("All demos completed successfully!")
        
    except Exception as e:
        print(f"Error during demo execution: {e}")
        print("Please check your Azure OpenAI configuration and try again.")

if __name__ == "__main__":
    main()
