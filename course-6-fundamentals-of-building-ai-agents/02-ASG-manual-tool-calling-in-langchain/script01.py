"""
LCEL RunnableParallel Demonstration
================================================================================

Module: script01.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates the simplest usage of LangChain Expression Language (LCEL)
    RunnableParallel for running multiple operations in parallel.
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.7,
    max_tokens=500
)

def demo_basic_parallel():
    """Demonstrate basic RunnableParallel usage"""
    print("\n" + "="*60)
    print("Demo 1: Basic RunnableParallel")
    print("="*60)
    
    # Create different prompts for parallel execution
    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarize this text in one sentence: {text}"
    )
    
    sentiment_prompt = ChatPromptTemplate.from_template(
        "What is the sentiment of this text (positive/negative/neutral)? {text}"
    )
    
    keywords_prompt = ChatPromptTemplate.from_template(
        "Extract 3 key topics from this text: {text}"
    )
    
    # Create individual chains
    summarize_chain = summarize_prompt | llm | StrOutputParser()
    sentiment_chain = sentiment_prompt | llm | StrOutputParser()
    keywords_chain = keywords_prompt | llm | StrOutputParser()
    
    # Create parallel runnable
    parallel_chain = RunnableParallel(
        summary=summarize_chain,
        sentiment=sentiment_chain,
        keywords=keywords_chain
    )
    
    # Test text
    test_text = """
    Artificial Intelligence has revolutionized many industries and continues to grow rapidly. 
    Machine learning algorithms are becoming more sophisticated, enabling breakthrough 
    applications in healthcare, finance, and transportation. The future looks bright 
    for AI development and its positive impact on society.
    """
    
    print(f"Input text: {test_text.strip()}")
    print("\nRunning parallel analysis...")
    
    # Execute parallel operations
    result = parallel_chain.invoke({"text": test_text})
    
    print("\nResults:")
    print(f"Summary: {result['summary']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Keywords: {result['keywords']}")

def demo_parallel_with_functions():
    """Demonstrate RunnableParallel with simple functions"""
    print("\n" + "="*60)
    print("Demo 2: RunnableParallel with Functions")
    print("="*60)
    
    # Define simple processing functions
    def count_words(text: str) -> str:
        count = len(text.split())
        return f"Word count: {count}"
    
    def count_characters(text: str) -> str:
        count = len(text)
        return f"Character count: {count}"
    
    def count_sentences(text: str) -> str:
        count = text.count('.') + text.count('!') + text.count('?')
        return f"Sentence count: {count}"
    
    # Create RunnableLambda for each function
    word_counter = RunnableLambda(count_words)
    char_counter = RunnableLambda(count_characters)
    sentence_counter = RunnableLambda(count_sentences)
    
    # Create parallel runnable
    stats_analyzer = RunnableParallel(
        words=word_counter,
        characters=char_counter,
        sentences=sentence_counter
    )
    
    test_text = "Hello world! This is a test. How are you doing today?"
    
    print(f"Input text: '{test_text}'")
    print("\nRunning parallel text analysis...")
    
    result = stats_analyzer.invoke(test_text)
    
    print("\nText Statistics:")
    print(f"- {result['words']}")
    print(f"- {result['characters']}")
    print(f"- {result['sentences']}")

def demo_mixed_parallel():
    """Demonstrate mixing LLM calls and functions in parallel"""
    print("\n" + "="*60)
    print("Demo 3: Mixed Parallel Operations")
    print("="*60)
    
    # LLM-based analysis
    theme_prompt = ChatPromptTemplate.from_template(
        "What is the main theme of this text? {text}"
    )
    theme_chain = theme_prompt | llm | StrOutputParser()
    
    # Function-based analysis
    def text_stats(text: str) -> dict:
        return {
            "length": len(text),
            "words": len(text.split()),
            "uppercase_chars": sum(1 for c in text if c.isupper())
        }
    
    stats_function = RunnableLambda(text_stats)
    
    # Create parallel chain mixing LLM and function
    mixed_analyzer = RunnableParallel(
        theme=theme_chain,
        stats=stats_function
    )
    
    test_text = "Technology is reshaping our world at an unprecedented pace."
    
    print(f"Input text: '{test_text}'")
    print("\nRunning mixed parallel analysis...")
    
    result = mixed_analyzer.invoke({"text": test_text})
    
    print("\nAnalysis Results:")
    print(f"Theme: {result['theme']}")
    print(f"Statistics: {result['stats']}")

def demo_chained_parallel():
    """Demonstrate chaining with parallel operations"""
    print("\n" + "="*60)
    print("Demo 4: Chained Parallel Operations")
    print("="*60)
    
    # First stage: parallel preprocessing
    def to_uppercase(text: str) -> str:
        return text.upper()
    
    def to_lowercase(text: str) -> str:
        return text.lower()
    
    def reverse_text(text: str) -> str:
        return text[::-1]
    
    # Create parallel preprocessing
    preprocessor = RunnableParallel(
        upper=RunnableLambda(to_uppercase),
        lower=RunnableLambda(to_lowercase),
        reversed=RunnableLambda(reverse_text)
    )
    
    # Second stage: process results
    def combine_results(results: dict) -> str:
        return f"Upper: {results['upper'][:20]}...\nLower: {results['lower'][:20]}...\nReversed: {results['reversed'][:20]}..."
    
    # Chain preprocessing with combination
    full_chain = preprocessor | RunnableLambda(combine_results)
    
    test_text = "LangChain Expression Language is powerful!"
    
    print(f"Input text: '{test_text}'")
    print("\nRunning chained parallel operations...")
    
    result = full_chain.invoke(test_text)
    
    print("\nProcessed Results:")
    print(result)

def run_all_demos():
    """Run all RunnableParallel demonstrations"""
    print("\n" + "="*80)
    print("LCEL RunnableParallel Demonstrations")
    print("="*80)
    
    try:
        demo_basic_parallel()
        demo_parallel_with_functions()
        demo_mixed_parallel()
        demo_chained_parallel()
        
        print("\n" + "="*80)
        print("All demonstrations completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"Error during demonstration: {e}")

if __name__ == "__main__":
    run_all_demos() 