import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Model 1: GPT-4.1-mini
model1 = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_deployment="gpt-4.1-mini",
    temperature=0.7,
    max_tokens=1000
)

# Model 2: GPT-4.1 
model2 = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_deployment="gpt-4.1", 
    temperature=0.7,
    max_tokens=1000
)

# Judge Model: o3
judge_model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_deployment="o3",
)

print("=== Benchmark: Creative Writing Marketing Posts - Funniness Evaluation ===\n")

# Test prompt for creative writing marketing posts
test_prompt = """Write a funny marketing post for a fictional product called 'Invisible Socks' - socks that are so comfortable you forget you're wearing them. Make it humorous and engaging for social media."""

# Get responses from both models
print("🤖 Getting response from GPT-4.1-mini...")
messages = [
    SystemMessage(content="You are a creative marketing copywriter known for your humor and wit."),
    HumanMessage(content=test_prompt)
]

response1 = model1.invoke(messages)
print(f"GPT-4.1-mini Response:\n{response1.content}\n")
print("-" * 80)

print("🤖 Getting response from GPT-4.1...")
response2 = model2.invoke(messages)
print(f"GPT-4.1 Response:\n{response2.content}\n")
print("-" * 80)

# Judge evaluation
print("⚖️ Getting evaluation from o3 judge...")
judge_prompt = f"""
You are an expert judge evaluating creative writing for marketing posts. Your task is to determine which response is FUNNIER and more engaging for social media marketing.

RESPONSE A (GPT-4.1-mini):
{response1.content}

RESPONSE B (GPT-4.1):
{response2.content}

Evaluate both responses based on:
1. Humor level (how funny is it?)
2. Creativity and originality
3. Marketing effectiveness
4. Social media engagement potential

Please provide:
1. A brief analysis of each response
2. Your final judgment on which is funnier
3. A score out of 10 for each response
4. The winner

Format your response clearly with sections for each evaluation criteria.
"""

judge_messages = [
    SystemMessage(content="You are an expert judge specializing in evaluating humor and creativity in marketing content."),
    HumanMessage(content=judge_prompt)
]

judge_response = judge_model.invoke(judge_messages)
print(f"Judge Evaluation:\n{judge_response.content}")

print("\n" + "=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)