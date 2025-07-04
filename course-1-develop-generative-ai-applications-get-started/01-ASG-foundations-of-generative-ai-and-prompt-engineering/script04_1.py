"""
LangChain Expression Language (LCEL) Demo with Real Model - Based on Official Documentation
Reference: https://python.langchain.com/docs/concepts/lcel/

This demonstrates the core concepts of LCEL including:
- Runnable interface and composition primitives
- Real Azure OpenAI model integration
- Type coercion (dictionary → RunnableParallel, function → RunnableLambda)
- The | operator for chaining
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
            max_tokens=500
        )
        return llm
    except Exception as e:
        print(f"Error setting up Azure OpenAI: {e}")
        print("Please ensure all required environment variables are set.")
        return None

llm = setup_azure_openai()

print("===== llm.invoke output =====")
print(llm.invoke("What is the capital of France?"))

parser = StrOutputParser()

prompt_template = PromptTemplate.from_template("What is the capital of {country}?")
chain = prompt_template | llm | parser

print("\n===== chain.invoke output =====")
output = chain.invoke({"country": "France"})
print(output)

print("\n===== chain.batch output =====")
output = chain.batch([{"country": "France"}, {"country": "Germany"}])
print(output)