"""
Pandas Agent Demonstration
================================================================================

Module: script01.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates the simplest usage of LangChain's pandas agent
    for querying and analyzing pandas dataframes using natural language.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

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

def create_sample_data():
    """Create sample dataframes for demonstration"""
    print("Creating sample datasets...")
    
    # Sales data
    sales_data = {
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Laptop'],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics'],
        'price': [1200, 25, 75, 300, 1200],
        'quantity': [2, 10, 5, 1, 1],
        'sales_rep': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob']
    }
    
    # Customer data
    customer_data = {
        'customer_id': [1, 2, 3, 4, 5],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
        'age': [35, 28, 42, 31, 39],
        'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
        'total_spent': [2400, 25, 75, 300, 1200]
    }
    
    sales_df = pd.DataFrame(sales_data)
    customer_df = pd.DataFrame(customer_data)
    
    # Convert date column to datetime
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    
    # Calculate total sales amount
    sales_df['total_amount'] = sales_df['price'] * sales_df['quantity']
    
    print("Sample datasets created successfully!")
    return sales_df, customer_df

def demo_basic_pandas_agent():
    """Demonstrate basic pandas agent functionality"""
    print("\n" + "="*60)
    print("Demo 1: Basic Pandas Agent")
    print("="*60)
    
    # Create sample data
    sales_df, _ = create_sample_data()
    
    print("Sales DataFrame:")
    print(sales_df)
    print()
    
    # Create pandas agent
    agent = create_pandas_dataframe_agent(
        llm,
        sales_df,
        agent_type="tool-calling",
        verbose=True,
        allow_dangerous_code=True
    )
    
    # Sample queries
    queries = [
        "What is the shape of this dataframe?",
        "What are the column names?",
        "What is the total revenue (sum of total_amount)?",
        "Which product has the highest price?",
        "How many unique products are there?"
    ]
    
    print("Running queries against the sales dataframe...")
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        try:
            result = agent.invoke(query)
            print(f"Answer: {result['output']}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)

def demo_data_analysis():
    """Demonstrate data analysis capabilities"""
    print("\n" + "="*60)
    print("Demo 2: Data Analysis with Pandas Agent")
    print("="*60)
    
    sales_df, customer_df = create_sample_data()
    
    print("Sales DataFrame:")
    print(sales_df.head())
    print()
    
    # Create agent for sales analysis
    sales_agent = create_pandas_dataframe_agent(
        llm,
        sales_df,
        agent_type="tool-calling",
        verbose=True,
        allow_dangerous_code=True
    )
    
    # Analysis queries
    analysis_queries = [
        "What is the average price of products?",
        "Group by sales_rep and show total sales amount for each",
        "What is the date range of the sales data?",
        "Which sales rep has the highest total sales?",
        "Calculate the total quantity sold"
    ]
    
    print("Running analysis queries...")
    
    for i, query in enumerate(analysis_queries, 1):
        print(f"\n--- Analysis {i}: {query} ---")
        try:
            result = sales_agent.invoke(query)
            print(f"Result: {result['output']}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)

def demo_multiple_dataframes():
    """Demonstrate working with multiple dataframes"""
    print("\n" + "="*60)
    print("Demo 3: Multiple DataFrames")
    print("="*60)
    
    sales_df, customer_df = create_sample_data()
    
    print("Sales DataFrame:")
    print(sales_df.head())
    print("\nCustomer DataFrame:")
    print(customer_df.head())
    print()
    
    # Create agent with multiple dataframes
    agent = create_pandas_dataframe_agent(
        llm,
        [sales_df, customer_df],
        agent_type="tool-calling",
        verbose=True,
        allow_dangerous_code=True
    )
    
    # Queries involving multiple dataframes
    multi_queries = [
        "How many dataframes are available?",
        "What are the shapes of both dataframes?",
        "Compare the number of rows between the two dataframes",
        "What columns are available in each dataframe?"
    ]
    
    print("Running queries on multiple dataframes...")
    
    for i, query in enumerate(multi_queries, 1):
        print(f"\n--- Multi-DataFrame Query {i}: {query} ---")
        try:
            result = agent.invoke(query)
            print(f"Answer: {result['output']}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)

def demo_statistical_analysis():
    """Demonstrate statistical analysis capabilities"""
    print("\n" + "="*60)
    print("Demo 4: Statistical Analysis")
    print("="*60)
    
    sales_df, _ = create_sample_data()
    
    # Create agent for statistical analysis
    stats_agent = create_pandas_dataframe_agent(
        llm,
        sales_df,
        agent_type="tool-calling",
        verbose=True,
        allow_dangerous_code=True
    )
    
    # Statistical queries
    stats_queries = [
        "Show descriptive statistics for numerical columns",
        "What is the correlation between price and quantity?",
        "Calculate the median total_amount",
        "Show the distribution of products (value counts)",
        "What is the standard deviation of prices?"
    ]
    
    print("Running statistical analysis queries...")
    
    for i, query in enumerate(stats_queries, 1):
        print(f"\n--- Statistical Query {i}: {query} ---")
        try:
            result = stats_agent.invoke(query)
            print(f"Result: {result['output']}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)

def run_all_demos():
    """Run all pandas agent demonstrations"""
    print("\n" + "="*80)
    print("PANDAS AGENT DEMONSTRATIONS")
    print("="*80)
    
    try:
        demo_basic_pandas_agent()
        demo_data_analysis()
        demo_multiple_dataframes()
        demo_statistical_analysis()
        
        print("\n" + "="*80)
        print("All pandas agent demonstrations completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_demos() 