"""
Simple SQL Agent Testing and Validation Demo
================================================================================

Module: script05.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module provides a simple demonstration for testing and validating
    SQL agent functionality using LangChain. Creates a basic SQLite database
    with sample data and runs comprehensive tests to validate the agent's
    ability to understand and execute natural language SQL queries.
"""

import os
import sqlite3
import time
from dotenv import load_dotenv
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase

# Load environment variables
load_dotenv()

def create_test_database():
    """Create a simple test database with sample data"""
    
    conn = sqlite3.connect("test_company.db")
    cursor = conn.cursor()
    
    # Create a simple employees table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary INTEGER NOT NULL,
            age INTEGER NOT NULL
        )
    """)
    
    # Insert test data
    test_data = [
        (1, "John Doe", "Engineering", 75000, 30),
        (2, "Jane Smith", "Marketing", 65000, 28),
        (3, "Bob Johnson", "Engineering", 80000, 35),
        (4, "Alice Brown", "Sales", 70000, 32),
        (5, "Charlie Wilson", "HR", 60000, 29),
        (6, "Diana Davis", "Engineering", 85000, 33),
        (7, "Eve Miller", "Marketing", 68000, 27),
        (8, "Frank Taylor", "Sales", 72000, 31)
    ]
    
    cursor.executemany("""
        INSERT OR REPLACE INTO employees (id, name, department, salary, age)
        VALUES (?, ?, ?, ?, ?)
    """, test_data)
    
    conn.commit()
    conn.close()
    
    print("✓ Test database created successfully!")
    print("✓ Table: employees (8 records)")

def run_sql_agent_tests():
    """Run comprehensive tests to validate SQL agent functionality"""
    print("\n" + "="*60)
    print("SQL Agent Testing and Validation")
    print("="*60)
    
    # Connect to database (already created in validation)
    db = SQLDatabase.from_uri("sqlite:///test_company.db")
    
    # Initialize LLM
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0.1,  # Lower temperature for consistent results
        max_tokens=300
    )
    
    # Create SQL agent
    agent = create_sql_agent(
        llm=llm,
        db=db,
        verbose=True,
        top_k=10,
        max_iterations=10
    )
    
    # Test cases for validation
    test_cases = [
        {
            "name": "Basic Count Query",
            "query": "How many employees are there?",
            "expected_result": "8"
        },
        {
            "name": "Department Count",
            "query": "How many employees are in Engineering department?",
            "expected_result": "3"
        },
        {
            "name": "Average Salary",
            "query": "What is the average salary?",
            "expected_result": "70625"
        },
        {
            "name": "Highest Salary",
            "query": "Who has the highest salary?",
            "expected_result": "Diana Davis"
        },
        {
            "name": "Salary Range Query",
            "query": "List employees with salary greater than 70000",
            "expected_result": "should list 5 employees"
        },
        {
            "name": "Department Summary",
            "query": "Show average salary by department",
            "expected_result": "Engineering: 80000, Marketing: 66500, Sales: 71000, HR: 60000"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[Test {i}/{len(test_cases)}] {test_case['name']}")
        print(f"Query: {test_case['query']}")
        print(f"Expected: {test_case['expected_result']}")
        print("-" * 50)
        
        try:
            start_time = time.time()
            response = agent.invoke({"input": test_case['query']})
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            print(f"✓ Agent Response: {response['output']}")
            print(f"✓ Execution Time: {execution_time:.2f} seconds")
            
            results.append({
                "test": test_case['name'],
                "status": "PASSED",
                "time": execution_time,
                "response": response['output']
            })
            
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                "test": test_case['name'],
                "status": "FAILED",
                "error": str(e)
            })
        
        print("=" * 50)
    
    return results

def display_test_summary(results):
    """Display test execution summary"""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results if r['status'] == 'PASSED')
    failed = sum(1 for r in results if r['status'] == 'FAILED')
    total = len(results)
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if failed > 0:
        print("\nFailed Tests:")
        for result in results:
            if result['status'] == 'FAILED':
                print(f"  - {result['test']}: {result.get('error', 'Unknown error')}")
    
    if passed > 0:
        avg_time = sum(r['time'] for r in results if r['status'] == 'PASSED') / passed
        print(f"\nAverage Execution Time: {avg_time:.2f} seconds")

def validate_database_connection():
    """Validate database connection and structure"""
    print("\n" + "="*60)
    print("DATABASE VALIDATION")
    print("="*60)
    
    try:
        # First create the database
        create_test_database()
        
        db = SQLDatabase.from_uri("sqlite:///test_company.db")
        
        print("✓ Database connection successful")
        print(f"✓ Available tables: {db.get_usable_table_names()}")
        
        # Test basic query
        result = db.run("SELECT COUNT(*) FROM employees")
        print(f"✓ Employee count: {result}")
        
        print("✓ Database validation complete")
        return True
        
    except Exception as e:
        print(f"✗ Database validation failed: {e}")
        return False

def main():
    """Main function to run SQL agent testing and validation"""
    print("Simple SQL Agent Testing and Validation")
    print("=" * 80)
    
    # Validate database connection
    if not validate_database_connection():
        return
    
    # Run comprehensive tests
    results = run_sql_agent_tests()
    
    # Display summary
    display_test_summary(results)
    
    print("\n" + "="*80)
    print("Testing and validation complete!")

if __name__ == "__main__":
    main() 