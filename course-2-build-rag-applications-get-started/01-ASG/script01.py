import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

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

