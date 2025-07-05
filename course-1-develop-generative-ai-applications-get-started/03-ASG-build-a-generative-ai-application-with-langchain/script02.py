import os
import json
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

# Simulated prompt configuration that can be updated after deployment
# In production, this would come from a database, config service, or external file
PROMPT_CONFIG = {
    "version": "1.0",
    "system_prompts": {
        "customer_service": {
            "v1": "You are a helpful customer service representative. Be polite and professional.",
            "v2": "You are an enthusiastic customer service expert. Use emojis and show excitement to help customers! 🎉"
        },
        "technical_support": {
            "v1": "You are a technical support specialist. Provide clear, step-by-step solutions.",
            "v2": "You are a senior technical guru. Explain complex concepts simply and always provide alternative solutions."
        }
    },
    "active_versions": {
        "customer_service": "v1",
        "technical_support": "v1"
    }
}

def get_system_prompt(service_type):
    """
    Retrieve the active system prompt for a given service type.
    This simulates how prompts can be updated after deployment.
    """
    active_version = PROMPT_CONFIG["active_versions"][service_type]
    return PROMPT_CONFIG["system_prompts"][service_type][active_version]

def update_prompt_version(service_type, version):
    """
    Update the active prompt version for a service type.
    This simulates updating prompts after deployment without code changes.
    """
    PROMPT_CONFIG["active_versions"][service_type] = version
    print(f"✅ Updated {service_type} to use prompt version {version}")

def demonstrate_ai_response(service_type, user_query):
    """
    Demonstrate how the AI responds with current prompt configuration.
    """
    system_prompt = get_system_prompt(service_type)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]
    
    response = llm.invoke(messages)
    
    print(f"\n--- {service_type.upper()} Response ---")
    print(f"Active Prompt Version: {PROMPT_CONFIG['active_versions'][service_type]}")
    print(f"System Prompt: {system_prompt}")
    print(f"User Query: {user_query}")
    print(f"AI Response: {response.content}")
    print("-" * 50)

# Demo: Prompt Updates After Deployment
print("🚀 DEMO: Prompt Updates After Deployment in Generative AI Applications")
print("=" * 70)

# Test query for customer service
customer_query = "I'm having trouble with my recent order. Can you help me?"

# Test query for technical support  
tech_query = "My application keeps crashing when I try to upload files. What should I do?"

print("\n📋 PHASE 1: Initial Deployment (Version 1 Prompts)")
print("=" * 50)

# Demonstrate initial responses
demonstrate_ai_response("customer_service", customer_query)
demonstrate_ai_response("technical_support", tech_query)

print("\n🔄 PHASE 2: Prompt Update After Deployment (No Code Changes)")
print("=" * 50)

# Simulate updating prompts after deployment
update_prompt_version("customer_service", "v2")
update_prompt_version("technical_support", "v2")

print("\n📋 PHASE 3: Updated Responses (Same Code, Different Prompts)")
print("=" * 50)

# Demonstrate how responses change with updated prompts
demonstrate_ai_response("customer_service", customer_query)
demonstrate_ai_response("technical_support", tech_query)

print("\n💡 KEY TAKEAWAYS:")
print("1. Prompts are externalized from code and can be updated independently")
print("2. Same application logic produces different behaviors with different prompts")
print("3. No code deployment needed - just configuration updates")
print("4. Enables A/B testing, gradual rollouts, and quick fixes")
print("5. Critical for production AI systems that need rapid iteration")
