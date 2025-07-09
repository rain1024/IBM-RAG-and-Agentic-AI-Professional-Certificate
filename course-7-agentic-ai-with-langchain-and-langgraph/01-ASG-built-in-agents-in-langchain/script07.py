"""
LangChain SQL Agent Database Connector Demo
================================================================================

Module: script07.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates how to use database connectors in SQL agents using LangChain's
    SQLDatabase utility. Shows different connection methods, connector configurations,
    and how the database connector interacts with the SQL agent to execute queries.
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

def demo_database_connector():
    """Demonstrate database connector functionality"""
    print("\n" + "="*60)
    print("Demo: Database Connector in SQL Agent")
    print("="*60)
    
    # Create sample database
    create_sample_database()
    
    # Method 1: Connect using URI string
    print("\n1. Connecting using URI string:")
    print("-" * 40)
    db_uri = "sqlite:///sample_company.db"
    print(f"Database URI: {db_uri}")
    
    db = SQLDatabase.from_uri(db_uri)
    print(f"Database type: {db.dialect}")
    print(f"Available tables: {db.get_table_names()}")
    
    # Method 2: Connect with custom configuration
    print("\n2. Connecting with custom configuration:")
    print("-" * 40)
    
    # Create database with custom settings
    db_custom = SQLDatabase.from_uri(
        db_uri,
        include_tables=['employees', 'departments'],  # Specify which tables to include
        sample_rows_in_table_info=3,  # Number of sample rows in table info
        max_string_length=100  # Maximum string length
    )
    
    print(f"Custom connector - Included tables: {db_custom.get_table_names()}")
    print(f"Custom connector - Table info preview:")
    print(db_custom.get_table_info())
    
    # Method 3: Show connector capabilities
    print("\n3. Database Connector Capabilities:")
    print("-" * 40)
    
    # Show available methods
    print("Available connector methods:")
    methods = [method for method in dir(db) if not method.startswith('_')]
    for method in methods[:10]:  # Show first 10 methods
        print(f"  - {method}")
    
    # Test direct SQL execution through connector
    print("\n4. Direct SQL execution through connector:")
    print("-" * 40)
    
    test_query = "SELECT COUNT(*) as total_employees FROM employees"
    result = db.run(test_query)
    print(f"Query: {test_query}")
    print(f"Result: {result}")
    
    # Show table structure
    print("\n5. Table structure information:")
    print("-" * 40)
    
    table_info = db.get_table_info(['employees'])
    print("Employees table structure:")
    print(table_info)

def demo_sql_agent_with_connector():
    """Demonstrate SQL agent using database connector"""
    print("\n" + "="*60)
    print("Demo: SQL Agent with Database Connector")
    print("="*60)
    
    # Connect to the database using connector
    db = SQLDatabase.from_uri(
        "sqlite:///sample_company.db",
        include_tables=['employees', 'departments'],
        sample_rows_in_table_info=2
    )
    
    print(f"Connector established for: {db.dialect}")
    print(f"Tables available through connector: {db.get_table_names()}")
    
    # Initialize Azure OpenAI LLM
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0.7,
        max_tokens=500
    )
    
    # Create SQL agent with database connector
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,  # Pass the configured database connector
        verbose=True,
        top_k=10,
        max_iterations=15
    )
    
    # Test queries to demonstrate connector usage
    queries = [
        "How many employees work in each department?",
        "What is the average salary by department?",
        "Show me the department with the highest budget",
        "List all employees in Engineering department"
    ]
    
    print("\nTesting SQL Agent with Database Connector:")
    print("-" * 50)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Natural Language Query: {query}")
        print("-" * 30)
        
        try:
            response = agent_executor.invoke({"input": query})
            print(f"Agent Response: {response['output']}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("=" * 50)

def demo_connector_error_handling():
    """Demonstrate error handling with database connector"""
    print("\n" + "="*60)
    print("Demo: Database Connector Error Handling")
    print("="*60)
    
    # Test connection to non-existent database
    print("\n1. Testing connection to non-existent database:")
    print("-" * 40)
    
    try:
        db_invalid = SQLDatabase.from_uri("sqlite:///nonexistent.db")
        print("Connection successful (empty database created)")
        print(f"Tables: {db_invalid.get_table_names()}")
    except Exception as e:
        print(f"Connection error: {e}")
    
    # Test invalid SQL query
    print("\n2. Testing invalid SQL query through connector:")
    print("-" * 40)
    
    db = SQLDatabase.from_uri("sqlite:///sample_company.db")
    try:
        result = db.run("SELECT * FROM invalid_table")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Query error: {e}")
    
    # Test connector with restricted tables
    print("\n3. Testing connector with restricted tables:")
    print("-" * 40)
    
    db_restricted = SQLDatabase.from_uri(
        "sqlite:///sample_company.db",
        include_tables=['employees']  # Only include employees table
    )
    
    print(f"Restricted tables: {db_restricted.get_table_names()}")
    
    try:
        result = db_restricted.run("SELECT * FROM departments LIMIT 1")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Restricted access error: {e}")

def main():
    """Main function to run the database connector demo"""
    print("LangChain SQL Agent Database Connector Demo")
    print("=" * 80)
    
    # Demo 1: Database connector functionality
    demo_database_connector()
    
    # Demo 2: SQL agent with connector
    demo_sql_agent_with_connector()
    
    # Demo 3: Error handling
    demo_connector_error_handling()
    
    print("\n" + "="*80)
    print("Demo completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main() 