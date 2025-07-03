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

print("=== LangChain Expression Language (LCEL) with Real Model Demo ===\n")

# Step 1: Basic Runnable concepts
print("1. LCEL Core Concept:")
print("   - Everything in LCEL is a 'Runnable' that implements the Runnable interface")
print("   - Chains are built by composing Runnables together")
print("   - Main composition primitives: RunnableSequence & RunnableParallel\n")

# Step 2: Setup real Azure OpenAI model
print("2. Setting up Real Azure OpenAI Model:")
model = setup_azure_openai()
if model:
    print(f"   ✓ Model initialized: {type(model)}")
    print(f"   ✓ Model is a Runnable: {hasattr(model, 'invoke')}")
else:
    print("   ✗ Model setup failed - using mock model instead")
    # Fallback to mock model
    def mock_model_func(prompt_value):
        if hasattr(prompt_value, 'text'):
            return f"Mock response to: {prompt_value.text[:50]}..."
        else:
            return f"Mock response to: {str(prompt_value)[:50]}..."
    model = RunnableLambda(mock_model_func)

print()

# Step 3: Template with variables (official pattern)
template = """You are a helpful assistant that provides clear explanations.

Topic: {topic}
Question: {question}

Please provide a clear and concise explanation."""

prompt = PromptTemplate.from_template(template)
print("3. PromptTemplate (a Runnable):")
print(f"   Template: {template}")
print(f"   Input variables: {prompt.input_variables}\n")

# Step 4: Output parser
parser = StrOutputParser()
print("4. Output Parser (a Runnable):")
print(f"   Parser: {parser}")
print(f"   Purpose: Converts model output to string\n")

# Step 5: Demonstrate the | operator with real model
print("5. The | Operator - Core LCEL Pattern with Real Model:")
print("   chain = prompt | model | parser  # Creates RunnableSequence automatically")

# Using | operator - this creates RunnableSequence automatically
chain = prompt | model | parser
print(f"   Chain created: {type(chain)}")

# Test the chain with real model
test_input = {"topic": "Python Lists", "question": "What are Python lists and how do they work?"}
print(f"   Input: {test_input}")

try:
    print("   Invoking chain with real model...")
    result = chain.invoke(test_input)
    print(f"   Real Model Result: {result}\n")
except Exception as e:
    print(f"   Error with real model: {e}")
    print("   This is expected if Azure OpenAI credentials are not configured\n")

# Step 6: Demonstrate batch processing (LCEL benefit)
print("6. LCEL Batch Processing (Real Model Benefit):")
batch_inputs = [
    {"topic": "Machine Learning", "question": "What is supervised learning?"},
    {"topic": "Data Science", "question": "What is the difference between statistics and data science?"}
]

try:
    print("   Processing multiple inputs in batch...")
    batch_results = chain.batch(batch_inputs)
    for i, result in enumerate(batch_results):
        print(f"   Batch {i+1}: {result[:100]}...")
    print()
except Exception as e:
    print(f"   Batch processing error: {e}")
    print("   This is expected if Azure OpenAI credentials are not configured\n")

# Step 7: Demonstrate dictionary auto-conversion to RunnableParallel
print("7. Automatic Type Coercion - Dictionary → RunnableParallel:")

def extract_topic(inputs):
    """Extract and format topic"""
    return f"Topic analysis: {inputs['topic']}"

def extract_question(inputs):
    """Extract and format question"""
    return f"Question analysis: {inputs['question']}"

# Dictionary gets auto-converted to RunnableParallel when used in chain
mapping = {
    "topic_info": extract_topic,  # Function auto-converts to RunnableLambda
    "question_info": extract_question
}

print("   Dictionary with functions:")
print(f"   {mapping}")

# This mapping gets auto-converted when used in a chain
try:
    auto_chain = mapping | RunnableLambda(lambda x: f"Combined analysis: {x}")
    auto_result = auto_chain.invoke(test_input)
    print(f"   Auto-coercion result: {auto_result}\n")
except Exception as e:
    print(f"   Auto-coercion demo error: {e}\n")

# Step 8: Advanced LCEL pattern with real model
print("8. Advanced LCEL Pattern - Parallel + Sequential:")
try:
    # Create a parallel step that processes input in multiple ways
    parallel_step = {
        "original_question": lambda x: x,  # Pass through
        "topic_context": lambda x: f"Context: {x['topic']} is an important concept in programming."
    }
    
    # Create a chain that uses parallel processing then sends to model
    advanced_template = """Based on the context: {topic_context}

Original question: {original_question}

Please provide a detailed explanation."""
    
    advanced_prompt = PromptTemplate.from_template(advanced_template)
    advanced_chain = parallel_step | advanced_prompt | model | parser
    
    print("   Advanced chain: parallel → prompt → model → parser")
    if hasattr(model, 'azure_endpoint'):  # Check if real model
        advanced_result = advanced_chain.invoke(test_input)
        print(f"   Advanced result: {advanced_result[:150]}...")
    else:
        print("   (Skipping advanced demo - mock model in use)")
    print()
except Exception as e:
    print(f"   Advanced pattern error: {e}\n")

# Step 9: LCEL Benefits demonstration
print("9. LCEL Benefits (from official docs):")
print("   ✓ Optimized parallel execution")
print("   ✓ Guaranteed async support")
print("   ✓ Simplified streaming")
print("   ✓ Seamless LangSmith tracing")
print("   ✓ Standard API across all chains")
print("   ✓ Deployable with LangServe\n")

# Step 10: When to use LCEL vs LangGraph
print("10. When to Use LCEL (Official Guidelines):")
print("   ✓ Simple chains (prompt + llm + parser)")
print("   ✓ Simple retrieval setups")
print("   ✗ Complex state management → Use LangGraph")
print("   ✗ Branching, cycles, multiple agents → Use LangGraph")
print("   ✗ Single LLM call → Call model directly\n")

# Step 11: Demonstrate typical LCEL pattern components
print("11. Typical LCEL Pattern Components:")
print("   1. ✓ Template with variables in curly braces: {variable}")
print("   2. ✓ PromptTemplate instance creation")
print("   3. ✓ Pipe operator (|) for chaining components")
print("   4. ✗ Manual pre-processing (NOT needed - automatic type coercion!)\n")

# Step 12: Quiz answer based on official documentation
print("=== Quiz Answer (Based on Official LCEL Docs) ===")
print("Question: Which step is NOT part of creating a typical LCEL pattern?")
print("")
print("The four options were:")
print("1. ✓ Defining a template with variables in curly braces")
print("2. ✓ Creating a PromptTemplate instance from the template")
print("3. ✗ Pre-processing all input data to ensure type compatibility")
print("4. ✓ Using the pipe operator to connect components into a chain")
print("")
print("ANSWER: 'Pre-processing all input data to ensure type compatibility'")
print("")
print("WHY: LCEL provides automatic type coercion:")
print("- Dictionary → RunnableParallel (automatically)")
print("- Function → RunnableLambda (automatically)")
print("- No manual pre-processing required!")
print("")
print("Reference: https://python.langchain.com/docs/concepts/lcel/")

print("\n=== Demo Complete ===")
print("The script demonstrates LCEL with a real Azure OpenAI model!")
if hasattr(model, 'azure_endpoint'):
    print("✓ Real model successfully integrated")
else:
    print("⚠ Mock model used (Azure OpenAI credentials not configured)")
print("LCEL handles type compatibility automatically - no manual pre-processing needed!")

