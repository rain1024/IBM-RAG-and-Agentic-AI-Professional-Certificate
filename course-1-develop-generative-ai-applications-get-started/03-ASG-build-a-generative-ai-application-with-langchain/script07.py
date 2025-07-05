"""
LangChain Demo with Azure OpenAI
A simple demonstration of using LangChain with Azure OpenAI for prompt engineering.
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate

# Load environment variables
load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.7,
    max_tokens=1000
)

"""Demo 1: Simple direct prompt"""
print("\n=== Demo 1: Simple Direct Prompt ===")

messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Explain what machine learning is in simple terms.")
]

response = llm.invoke(messages)
print(f"Response: {response.content}")


"""Demo 2: Using Few-Shot Prompting with FewShotPromptTemplate"""
print("\n=== Demo 2: Using Few-Shot Prompting ===")

# Define examples for sentiment analysis
examples = [
    {"text": "I love this product! It's amazing.", "sentiment": "positive"},
    {"text": "This is terrible. I hate it.", "sentiment": "negative"},
    {"text": "The weather is okay today.", "sentiment": "neutral"},
    {"text": "Best purchase I've ever made!", "sentiment": "positive"},
    {"text": "Waste of money. Very disappointed.", "sentiment": "negative"}
]

# Create a template for formatting each example
example_prompt = PromptTemplate(
    input_variables=["text", "sentiment"],
    template="Text: {text}\nSentiment: {sentiment}"
)

# Create the few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Classify the sentiment of the following texts as positive, negative, or neutral:",
    suffix="Text: {input}\nSentiment:",
    input_variables=["input"],
    example_separator="\n\n"
)

# Test with a new text
test_text = "This movie was absolutely fantastic! I enjoyed every minute of it."
prompt = few_shot_prompt.format(input=test_text)

print("Generated Few-Shot Prompt:")
print("-" * 50)
print(prompt)
print("-" * 50)

# Use the prompt with the LLM
messages = [
    SystemMessage(content="You are a sentiment analysis expert."),
    HumanMessage(content=prompt)
]

response = llm.invoke(messages)
print(f"\nLLM Response: {response.content}")