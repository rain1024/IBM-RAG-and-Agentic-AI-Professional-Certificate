"""
LangChain SQL Agent Demo
================================================================================

Module: script02.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates LangChain SQL agent functionality using create_sql_agent
    from langchain_cohere. Creates a simple SQLite database with sample data and
    shows how to query it using natural language.
"""

import os
import sqlite3
from dotenv import load_dotenv
from langchain_cohere import create_sql_agent
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase

# Load environment variables
load_dotenv()

def create_sample_database():
    """Create a simple SQLite database with sample data for demonstration"""
    
    # Create in-memory SQLite database
    conn = sqlite3.connect("sample_company.db")
    cursor = conn.cursor()
    
    # Create employees table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary INTEGER NOT NULL,
            hire_date TEXT NOT NULL
        )
    """)
    
    # Create departments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            manager TEXT NOT NULL,
            budget INTEGER NOT NULL
        )
    """)
    
    # Insert sample employees
    employees_data = [
        (1, "Alice Johnson", "Engineering", 85000, "2023-01-15"),
        (2, "Bob Smith", "Marketing", 65000, "2023-02-20"),
        (3, "Carol Davis", "Engineering", 90000, "2022-11-10"),
        (4, "David Wilson", "Sales", 70000, "2023-03-05"),
        (5, "Eva Brown", "HR", 60000, "2023-01-30"),
        (6, "Frank Miller", "Engineering", 95000, "2022-09-15"),
        (7, "Grace Lee", "Marketing", 68000, "2023-04-12"),
        (8, "Henry Taylor", "Sales", 75000, "2022-12-08")
    ]
    
    cursor.executemany("""
        INSERT OR REPLACE INTO employees (id, name, department, salary, hire_date)
        VALUES (?, ?, ?, ?, ?)
    """, employees_data)
    
    # Insert sample departments
    departments_data = [
        (1, "Engineering", "Carol Davis", 500000),
        (2, "Marketing", "Grace Lee", 300000),
        (3, "Sales", "Henry Taylor", 400000),
        (4, "HR", "Eva Brown", 200000)
    ]
    
    cursor.executemany("""
        INSERT OR REPLACE INTO departments (id, name, manager, budget)
        VALUES (?, ?, ?, ?)
    """, departments_data)
    
    conn.commit()
    conn.close()
    
    print("Sample database created successfully!")
    print("Tables: employees, departments")
    print("Sample data inserted.")

def demo_sql_agent():
    """Demonstrate SQL agent functionality"""
    print("\n" + "="*60)
    print("Demo: SQL Agent with Natural Language Queries")
    print("="*60)
    
    # Create sample database
    create_sample_database()
    
    # Connect to the database
    db = SQLDatabase.from_uri("sqlite:///sample_company.db")
    
    # Initialize Azure OpenAI LLM
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0.7,
        max_tokens=500
    )
    
    # Create SQL agent
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        verbose=True,
        top_k=10,
        max_iterations=15
    )
    
    # Test queries
    queries = [
        "How many employees are there in total?",
        "What is the average salary by department?",
        "Show me all employees in the Engineering department",
        "Who is the highest paid employee?",
        "What are the names of all departments and their managers?",
        "Find employees hired after 2023-01-01",
        "What is the total budget across all departments?"
    ]
    
    for query in queries:
        print(f"\nNatural Language Query: {query}")
        print("-" * 50)
        
        try:
            response = agent_executor.invoke({"input": query})
            print(f"Agent Response: {response['output']}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "="*50)

def show_database_schema():
    """Display the database schema for reference"""
    print("\n" + "="*60)
    print("Database Schema")
    print("="*60)
    
    # First create the database if it doesn't exist
    create_sample_database()
    
    db = SQLDatabase.from_uri("sqlite:///sample_company.db")
    print("Available tables:")
    print(db.get_table_names())
    
    print("\nTable info:")
    print(db.get_table_info())

def main():
    """Main function to run the SQL agent demo"""
    print("LangChain SQL Agent Demo")
    print("=" * 80)
    
    # Check if Azure OpenAI API key is set
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("ERROR: AZURE_OPENAI_API_KEY environment variable not set!")
        print("Please set your Azure OpenAI API key in the .env file")
        return
    
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("ERROR: AZURE_OPENAI_ENDPOINT environment variable not set!")
        print("Please set your Azure OpenAI endpoint in the .env file")
        return
    
    if not os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"):
        print("ERROR: AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set!")
        print("Please set your Azure OpenAI deployment name in the .env file")
        return
    
    # Show database schema
    show_database_schema()
    
    # Run the demo
    demo_sql_agent()

if __name__ == "__main__":
    main() 