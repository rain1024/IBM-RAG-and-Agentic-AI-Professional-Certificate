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
from langchain.output_parsers import CommaSeparatedListOutputParser

# Load environment variables
load_dotenv()

# Create the parser
parser = CommaSeparatedListOutputParser()

# Get format instructions from the parser
format_instructions = parser.get_format_instructions()
print("Format Instructions:")
print(format_instructions)
print()

# Set up Azure OpenAI client for the demo
llm_demo = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.7,
    max_tokens=100
)

print("==== Example 1 ====")
# Create prompt template with format instructions
system_message = SystemMessage(
    content="""
    You are a helpful assistant that provides lists of items.
    """)

human_message = HumanMessage(
    content="""
    "List 5 popular programming languages.\n{format_instructions}"
    """
)

prompt_demo = ChatPromptTemplate.from_messages([
    system_message, human_message
])

# Format the prompt with format instructions
formatted_prompt_demo = prompt_demo.format_messages(
    format_instructions=format_instructions
)

chain = prompt_demo | llm_demo | parser

output = chain.invoke({"format_instructions": format_instructions})

print("Output:\n", output)

print("==== Example 2 ====")

system_message = SystemMessage(
    content="""
    You are a helpful assistant that provides lists of items.
    """)

human_message = HumanMessage(
    content="""
    List 10 famous people in the world.
    {format_instructions}
    """
)

prompt_demo = ChatPromptTemplate.from_messages([
    system_message, human_message
])

# Format the prompt with format instructions
formatted_prompt_demo = prompt_demo.format_messages(
    format_instructions=format_instructions
)

chain = prompt_demo | llm_demo | parser

output = chain.invoke({"format_instructions": format_instructions})

print("Output:\n", output)
