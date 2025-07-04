#!/usr/bin/env python3
"""
Simple LCEL (LangChain Expression Language) Pattern Demonstration
Shows the 4-step sequence to create a functional LCEL pattern
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Set up the LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.7
)

print("=== 4-Step LCEL Pattern ===")
print()

# Step 1: Define a template with variables
print("Step 1: Define a template with variables")
template = "Tell me a joke about {topic}"
print(f"Template: {template}")
print()

# Step 2: Create a PromptTemplate
print("Step 2: Create a PromptTemplate")
prompt = ChatPromptTemplate.from_template(template)
print("PromptTemplate created")
print()

# Step 3: Build a chain using the pipe operator
print("Step 3: Build a chain using the pipe operator")
chain = prompt | llm
print("Chain: prompt | llm")
print()

# Step 4: Invoke with input values
print("Step 4: Invoke with input values")
topic = "love"
result = chain.invoke({"topic": topic})

print(f"Input: {topic}")
print(f"Output: {result.content}")
print()
print("LCEL Pattern Complete!")
print("Data flows: template → prompt → llm → result")