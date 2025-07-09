"""
LangChain Agent Tool Calling Demo - Random Number Generator
================================================================================

Module: script04.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates LangChain agent tool calling functionality.
    Uses create_tool_calling_agent and AgentExecutor with a random number generator tool.
"""

import os
import random
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from typing import List

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.7,
    max_tokens=1000
)

def generate_random_number(params: str) -> str:
    """
    Generate random numbers based on specified parameters.
    
    Args:
        params (str): Parameters in format "min,max,count,type" 
                     Example: "1,10,5,int" or "0.0,1.0,3,float"
                     Default: generates 1 integer between 1-100
        
    Returns:
        str: Generated random numbers in a formatted string
    """
    try:
        # Parse parameters or use defaults
        if params.strip():
            parts = params.split(',')
            min_val = float(parts[0]) if len(parts) > 0 else 1
            max_val = float(parts[1]) if len(parts) > 1 else 100
            count = int(parts[2]) if len(parts) > 2 else 1
            num_type = parts[3].strip() if len(parts) > 3 else 'int'
        else:
            min_val, max_val, count, num_type = 1, 100, 1, 'int'
        
        # Validate parameters
        if min_val >= max_val:
            return "Error: Giá trị min phải nhỏ hơn max"
        
        if count <= 0 or count > 100:
            return "Error: Số lượng phải từ 1 đến 100"
        
        # Generate random numbers
        numbers = []
        for _ in range(count):
            if num_type.lower() == 'float':
                num = random.uniform(min_val, max_val)
                numbers.append(f"{num:.2f}")
            else:
                num = random.randint(int(min_val), int(max_val))
                numbers.append(str(num))
        
        # Format output
        if count == 1:
            result = f"Số ngẫu nhiên được tạo: {numbers[0]}"
        else:
            result = f"Các số ngẫu nhiên được tạo ({count} số):\n"
            result += "\n".join([f"  - {num}" for num in numbers])
        
        result += f"\nTham số: Min={min_val}, Max={max_val}, Loại={num_type}"
        
        return result
        
    except Exception as e:
        return f"Error: Không thể tạo số ngẫu nhiên - {str(e)}"

def create_random_number_tool():
    """Create the random number generator tool for the agent"""
    return Tool(
        name="generate_random_number",
        description="""Generate random numbers based on parameters. 
        Format: 'min,max,count,type' where:
        - min: minimum value (default: 1)
        - max: maximum value (default: 100) 
        - count: number of values to generate (default: 1, max: 100)
        - type: 'int' for integers or 'float' for decimals (default: int)
        Examples: '1,10,5,int' or '0.0,1.0,3,float' or leave empty for default (1 integer 1-100)""",
        func=generate_random_number
    )

def demo_agent_tool_calling():
    """Demonstrate agent with random number generation tool"""
    print("\n" + "="*60)
    print("Demo: Agent Tool Calling với Random Number Generator")
    print("="*60)
    
    # Create tools
    tools = [create_random_number_tool()]
    
    # Create prompt template for agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là một trợ lý hữu ích có thể tạo số ngẫu nhiên.
        Khi người dùng yêu cầu tạo số ngẫu nhiên, hãy sử dụng tool generate_random_number.
        Luôn thân thiện và cung cấp phản hồi hữu ích.
        
        Hướng dẫn sử dụng tool:
        - Để tạo 1 số nguyên từ 1-100: không cần tham số hoặc dùng ""
        - Để tạo số với tham số cụ thể: "min,max,count,type"
        - Ví dụ: "1,10,5,int" tạo 5 số nguyên từ 1-10
        - Ví dụ: "0.0,1.0,3,float" tạo 3 số thập phân từ 0.0-1.0"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Test queries
    queries = [
        "Tạo cho tôi một số ngẫu nhiên",
        "Sinh 5 số ngẫu nhiên từ 1 đến 20",
        "Tôi cần 3 số thập phân từ 0 đến 1",
        "Tạo 10 số ngẫu nhiên từ 50 đến 100",
        "Chào bạn, bạn có khỏe không?",  # Non-random query to test general conversation
        "Hãy giúp tôi chọn một số may mắn từ 1 đến 49",
    ]
    
    for query in queries:
        print(f"\nCâu hỏi của người dùng: {query}")
        print("-" * 50)
        
        try:
            response = agent_executor.invoke({"input": query})
            print(f"Phản hồi của Agent: {response['output']}")
        except Exception as e:
            print(f"Lỗi: {e}")
        
        print("\n" + "="*50)

def main():
    """Main function to run the agent demo"""
    print("LangChain Agent Tool Calling Demo - Random Number Generator")
    print("=" * 80)
    
    # Run the demo
    demo_agent_tool_calling()

if __name__ == "__main__":
    main()