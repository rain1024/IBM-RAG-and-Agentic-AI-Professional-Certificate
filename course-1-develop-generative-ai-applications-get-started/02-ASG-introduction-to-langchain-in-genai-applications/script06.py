import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import time

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

def demo_without_memory():
    """Demonstrate chatbot without memory - loses context between calls"""
    print("\n" + "="*60)
    print("DEMO 1: CHATBOT WITHOUT MEMORY")
    print("="*60)
    print("This bot loses context between each interaction\n")
    
    # First interaction
    print("👤 User: Hi, my name is Alice. I'm 25 years old.")
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="Hi, my name is Alice. I'm 25 years old.")
    ]
    response = llm.invoke(messages)
    print(f"🤖 Bot: {response.content}")
    
    time.sleep(1)
    
    # Second interaction - bot has no memory of previous conversation
    print("\n👤 User: What is my name and age?")
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="What is my name and age?")
    ]
    response = llm.invoke(messages)
    print(f"🤖 Bot: {response.content}")
    
    time.sleep(1)
    
    # Third interaction
    print("\n👤 User: I told you my name earlier. Do you remember?")
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="I told you my name earlier. Do you remember?")
    ]
    response = llm.invoke(messages)
    print(f"🤖 Bot: {response.content}")
    
    print("\n❌ Result: The bot has no memory of previous conversations!")

def demo_with_memory():
    """Demonstrate chatbot with memory using LangGraph"""
    print("\n" + "="*60)
    print("DEMO 2: CHATBOT WITH MEMORY")
    print("="*60)
    print("This bot maintains context across interactions\n")
    
    # Create workflow with memory
    workflow = StateGraph(state_schema=MessagesState)
    
    def call_model(state: MessagesState):
        system_prompt = (
            "You are a helpful assistant. "
            "Answer all questions to the best of your ability."
        )
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": response}
    
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")
    
    # Add memory checkpointer
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    # Configuration for this conversation thread
    config = {"configurable": {"thread_id": "demo_conversation"}}
    
    # First interaction
    print("👤 User: Hi, my name is Alice. I'm 25 years old.")
    response = app.invoke(
        {"messages": [HumanMessage(content="Hi, my name is Alice. I'm 25 years old.")]},
        config=config
    )
    print(f"🤖 Bot: {response['messages'][-1].content}")
    
    time.sleep(1)
    
    # Second interaction - bot should remember previous conversation
    print("\n👤 User: What is my name and age?")
    response = app.invoke(
        {"messages": [HumanMessage(content="What is my name and age?")]},
        config=config
    )
    print(f"🤖 Bot: {response['messages'][-1].content}")
    
    time.sleep(1)
    
    # Third interaction
    print("\n👤 User: Can you tell me something about myself?")
    response = app.invoke(
        {"messages": [HumanMessage(content="Can you tell me something about myself?")]},
        config=config
    )
    print(f"🤖 Bot: {response['messages'][-1].content}")
    
    print("\n✅ Result: The bot remembers our entire conversation!")

def demo_memory_management():
    """Demonstrate memory management with message trimming"""
    print("\n" + "="*60)
    print("DEMO 3: MEMORY MANAGEMENT (MESSAGE TRIMMING)")
    print("="*60)
    print("This bot manages memory by keeping only recent messages\n")
    
    # Create workflow with trimming
    workflow = StateGraph(state_schema=MessagesState)
    
    # Define trimmer - keep only last 4 messages
    trimmer = trim_messages(strategy="last", max_tokens=4, token_counter=len)
    
    def call_model_with_trimming(state: MessagesState):
        trimmed_messages = trimmer.invoke(state["messages"])
        system_prompt = (
            "You are a helpful assistant. "
            "Answer all questions to the best of your ability."
        )
        messages = [SystemMessage(content=system_prompt)] + trimmed_messages
        response = llm.invoke(messages)
        return {"messages": response}
    
    workflow.add_node("model", call_model_with_trimming)
    workflow.add_edge(START, "model")
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    config = {"configurable": {"thread_id": "trimming_demo"}}
    
    # Create a longer conversation
    conversations = [
        "Hi, my name is Bob and I'm from New York.",
        "I work as a software engineer at a tech company.",
        "My favorite programming language is Python.",
        "I also enjoy playing guitar in my free time.",
        "I have a pet dog named Max.",
        "What do you remember about me?"
    ]
    
    for i, user_input in enumerate(conversations, 1):
        print(f"👤 User ({i}): {user_input}")
        response = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        print(f"🤖 Bot: {response['messages'][-1].content}")
        
        if i < len(conversations):
            time.sleep(1)
            print()
    
    print("\n⚠️  Result: The bot only remembers recent messages due to trimming!")

def main():
    """Run all demonstrations"""
    print("LangChain Chatbot Memory Demonstration")
    print("Based on: https://python.langchain.com/docs/how_to/chatbots_memory/")
    
    try:
        # Demo 1: Without memory
        demo_without_memory()
        
        # Demo 2: With memory
        demo_with_memory()
        
        # Demo 3: Memory management
        demo_memory_management()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("1. Without Memory: Each interaction is independent")
        print("2. With Memory: Full conversation history is maintained")
        print("3. Memory Management: Recent messages are kept, older ones discarded")
        print("\nMemory is crucial for creating conversational AI that feels natural!")
        
    except Exception as e:
        print(f"Error running demonstration: {e}")
        print("Make sure you have the required environment variables set in .env file")

if __name__ == "__main__":
    main()
