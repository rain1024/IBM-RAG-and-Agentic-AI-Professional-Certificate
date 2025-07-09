"""
LangChain Prompt Template Demo
================================================================================

Module: script06.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates LangChain Prompt Template functionality with LLM.
    Shows various ways to create, structure and use prompt templates.
    Based on: https://python.langchain.com/docs/concepts/prompt_templates/
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Dict, Any, List

# Load environment variables
load_dotenv()

# Initialize LLM using init_chat_model
llm = init_chat_model(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model_provider="azure_openai",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    temperature=0.1,
    max_tokens=1000
)

def demo_basic_prompt_template():
    """Demonstrate basic prompt template usage"""
    print("\n" + "="*60)
    print("Demo: Basic Prompt Template")
    print("="*60)
    
    # Simple string template
    template = "Tell me a {adjective} story about {subject} in {language}."
    prompt = PromptTemplate.from_template(template)
    
    print("Template:", template)
    print("-" * 40)
    
    # Format with different values
    examples = [
        {"adjective": "funny", "subject": "a robot", "language": "Vietnamese"},
        {"adjective": "mysterious", "subject": "an ancient castle", "language": "English"},
        {"adjective": "romantic", "subject": "two AI assistants", "language": "Vietnamese"}
    ]
    
    for example in examples:
        formatted_prompt = prompt.format(**example)
        print(f"Input: {example}")
        print(f"Formatted: {formatted_prompt}")
        
        # Get response from LLM
        response = llm.invoke(formatted_prompt)
        print(f"Response: {response.content[:100]}...")
        print("-" * 40)

def demo_chat_prompt_template():
    """Demonstrate ChatPromptTemplate with system and human messages"""
    print("\n" + "="*60)
    print("Demo: Chat Prompt Template")
    print("="*60)
    
    # Create chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful {role} expert. Answer questions professionally in {language}."),
        ("human", "I need help with: {question}")
    ])
    
    print("Chat Template Structure:")
    print("- System: You are a helpful {role} expert. Answer questions professionally in {language}.")
    print("- Human: I need help with: {question}")
    print("-" * 40)
    
    # Test different scenarios
    examples = [
        {
            "role": "Python programming",
            "language": "Vietnamese", 
            "question": "How to create a simple web scraper?"
        },
        {
            "role": "cooking",
            "language": "English",
            "question": "What's the best way to make pasta?"
        },
        {
            "role": "travel",
            "language": "Vietnamese",
            "question": "Best places to visit in Vietnam during summer?"
        }
    ]
    
    # Create chain
    chain = prompt | llm | StrOutputParser()
    
    for example in examples:
        print(f"Scenario: {example['role']} expert, {example['language']}")
        print(f"Question: {example['question']}")
        
        response = chain.invoke(example)
        print(f"Response: {response[:150]}...")
        print("-" * 40)

def demo_few_shot_prompt_template():
    """Demonstrate few-shot prompting with examples"""
    print("\n" + "="*60)
    print("Demo: Few-Shot Prompt Template")
    print("="*60)
    
    # Create few-shot examples for sentiment analysis
    examples = [
        {"text": "I love this product!", "sentiment": "positive"},
        {"text": "This is terrible quality.", "sentiment": "negative"},
        {"text": "It's okay, nothing special.", "sentiment": "neutral"},
        {"text": "Amazing experience, highly recommend!", "sentiment": "positive"},
        {"text": "Worst purchase ever made.", "sentiment": "negative"}
    ]
    
    # Build few-shot prompt
    example_prompt = PromptTemplate(
        input_variables=["text", "sentiment"],
        template="Text: {text}\nSentiment: {sentiment}"
    )
    
    # Create few-shot examples string
    few_shot_examples = "\n\n".join([
        example_prompt.format(**example) for example in examples
    ])
    
    # Main prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a sentiment analysis expert. Analyze the sentiment of text as positive, negative, or neutral.

Here are some examples:

{examples}

Now analyze the following text:"""),
        ("human", "Text: {text}\nSentiment:")
    ])
    
    print("Few-shot examples included in prompt:")
    for example in examples[:2]:  # Show first 2 examples
        print(f"Text: {example['text']} -> Sentiment: {example['sentiment']}")
    print("...")
    print("-" * 40)
    
    # Test on new text
    test_texts = [
        "This movie was absolutely fantastic!",
        "I hate waiting in long lines.",
        "The weather is fine today.",
        "Cảm ơn bạn rất nhiều, dịch vụ tuyệt vời!"
    ]
    
    chain = prompt | llm | StrOutputParser()
    
    for text in test_texts:
        response = chain.invoke({
            "examples": few_shot_examples,
            "text": text
        })
        print(f"Text: {text}")
        print(f"Analysis: {response.strip()}")
        print("-" * 40)

def demo_conditional_prompt_template():
    """Demonstrate conditional prompting based on input"""
    print("\n" + "="*60)
    print("Demo: Conditional Prompt Template")
    print("="*60)
    
    from langchain_core.runnables import RunnableLambda
    
    def create_conditional_prompt(inputs: Dict[str, Any]) -> ChatPromptTemplate:
        """Create different prompts based on user type"""
        user_type = inputs.get("user_type", "general")
        
        if user_type == "beginner":
            system_msg = "You are a patient teacher. Explain concepts simply with easy examples. Use Vietnamese if needed."
        elif user_type == "expert":
            system_msg = "You are talking to an expert. Be technical, concise, and use professional terminology."
        else:
            system_msg = "You are a helpful assistant. Provide balanced explanations suitable for general audience."
        
        return ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", "{question}")
        ])
    
    # Create conditional chain
    conditional_prompt = RunnableLambda(create_conditional_prompt)
    
    # Test cases
    test_cases = [
        {
            "user_type": "beginner",
            "question": "What is machine learning?"
        },
        {
            "user_type": "expert", 
            "question": "Explain gradient descent optimization"
        },
        {
            "user_type": "general",
            "question": "How does artificial intelligence work?"
        }
    ]
    
    for case in test_cases:
        print(f"User type: {case['user_type']}")
        print(f"Question: {case['question']}")
        
        # Get conditional prompt
        prompt = conditional_prompt.invoke(case)
        
        # Create chain and get response
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke(case)
        
        print(f"Response style: {case['user_type']} level")
        print(f"Response: {response[:200]}...")
        print("-" * 40)

def demo_structured_output_prompt():
    """Demonstrate prompts for structured output with Pydantic models"""
    print("\n" + "="*60)
    print("Demo: Structured Output with Prompt Template")
    print("="*60)
    
    # Define output structure
    class PersonInfo(BaseModel):
        name: str = Field(description="Person's full name")
        age: int = Field(description="Person's age in years") 
        occupation: str = Field(description="Person's job or profession")
        location: str = Field(description="Where the person lives")
        interests: List[str] = Field(description="List of person's hobbies or interests")
    
    # Create parser
    parser = JsonOutputParser(pydantic_object=PersonInfo)
    
    # Create prompt with format instructions
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at extracting structured information from text.
Extract the person's information and format it according to these instructions:

{format_instructions}"""),
        ("human", "Extract information about this person:\n\n{text}")
    ])
    
    # Partial with format instructions
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    
    print("Output format:")
    print(parser.get_format_instructions())
    print("-" * 40)
    
    # Test texts
    test_texts = [
        """John Smith is a 32-year-old software engineer living in San Francisco. 
        He enjoys hiking, reading sci-fi novels, and playing guitar in his spare time.""",
        
        """Maria Nguyen, 28 tuổi, là một bác sĩ ở Hà Nội. Cô ấy thích nấu ăn, 
        chụp ảnh và du lịch khám phá những địa điểm mới."""
    ]
    
    # Create chain
    chain = prompt | llm | parser
    
    for i, text in enumerate(test_texts, 1):
        print(f"Test {i}:")
        print(f"Input: {text}")
        
        try:
            result = chain.invoke({"text": text})
            print("Extracted info:")
            print(f"  Name: {result['name']}")
            print(f"  Age: {result['age']}")
            print(f"  Occupation: {result['occupation']}")
            print(f"  Location: {result['location']}")
            print(f"  Interests: {', '.join(result['interests'])}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 40)

def demo_message_placeholders():
    """Demonstrate using MessagesPlaceholder for dynamic conversation history"""
    print("\n" + "="*60)
    print("Demo: Message Placeholders for Conversation History")
    print("="*60)
    
    # Create prompt with message placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Keep track of conversation context."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    print("Prompt structure with MessagesPlaceholder:")
    print("- System message")
    print("- Chat history (dynamic)")
    print("- Current human input")
    print("-" * 40)
    
    # Simulate conversation history
    chat_history = [
        HumanMessage(content="My name is Alice and I'm learning Python."),
        AIMessage(content="Nice to meet you, Alice! Python is a great language to learn. What aspect of Python interests you most?"),
        HumanMessage(content="I want to learn about web development."),
        AIMessage(content="Great choice! For web development with Python, I'd recommend starting with Flask or Django frameworks.")
    ]
    
    # Current inputs
    current_inputs = [
        "Can you recommend some resources for learning Flask?",
        "What's the difference between Flask and Django?",
        "Should I learn HTML and CSS first?"
    ]
    
    chain = prompt | llm | StrOutputParser()
    
    for inp in current_inputs:
        print(f"Current question: {inp}")
        
        result = chain.invoke({
            "chat_history": chat_history,
            "input": inp
        })
        
        print(f"Response: {result}")
        
        # Add to history for next iteration
        chat_history.extend([
            HumanMessage(content=inp),
            AIMessage(content=result)
        ])
        
        print("-" * 40)

def main():
    """Main function to run all demos"""
    print("LangChain Prompt Template Demo")
    print("=" * 80)
    
    # Run all demos
    demo_basic_prompt_template()
    demo_chat_prompt_template()
    demo_few_shot_prompt_template()
    demo_conditional_prompt_template()
    demo_structured_output_prompt()
    demo_message_placeholders()
    
    print("\n" + "="*80)
    print("Demo completed!")

if __name__ == "__main__":
    main() 