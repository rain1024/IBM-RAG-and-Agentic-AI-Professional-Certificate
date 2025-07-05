import os
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

"""
Business Requirements Demonstration for Generative AI Applications
================================================================
This program demonstrates how generative AI can address real business needs
across different departments and use cases.
"""

def demonstrate_customer_service():
    """Demo 1: Customer Service Automation"""
    print("\n=== Demo 1: Customer Service Automation ===")
    print("Business Need: Automated customer support responses")
    
    messages = [
        SystemMessage(content="""You are a professional customer service representative for TechCorp, 
                      an e-commerce company. Respond helpfully and professionally to customer inquiries. 
                      Always maintain a friendly tone and offer solutions."""),
        HumanMessage(content="""I ordered a laptop 3 days ago but haven't received any shipping 
                     confirmation. My order number is #12345. When will it arrive?""")
    ]
    
    response = llm.invoke(messages)
    print(f"AI Response: {response.content}")
    print("Business Value: Reduces customer service workload, provides 24/7 support")

def demonstrate_content_marketing():
    """Demo 2: Marketing Content Generation"""
    print("\n=== Demo 2: Marketing Content Generation ===")
    print("Business Need: Create engaging marketing content quickly")
    
    messages = [
        SystemMessage(content="""You are a creative marketing copywriter. Create compelling, 
                      engaging content that drives customer action. Use persuasive language 
                      and highlight key benefits."""),
        HumanMessage(content="""Write a product description for a new wireless fitness tracker 
                     that monitors heart rate, steps, sleep, and has 7-day battery life. 
                     Target audience: health-conscious professionals aged 25-45.""")
    ]
    
    response = llm.invoke(messages)
    print(f"AI Generated Content: {response.content}")
    print("Business Value: Faster content creation, consistent brand voice, A/B testing capabilities")

def demonstrate_document_analysis():
    """Demo 3: Business Document Summarization"""
    print("\n=== Demo 3: Business Document Analysis ===")
    print("Business Need: Quickly analyze and summarize business documents")
    
    sample_report = """
    Q3 2024 Sales Report Summary:
    Total Revenue: $2.4M (15% increase from Q2)
    Top Performing Products: Wireless earbuds (+45%), Smart watches (+32%), Fitness trackers (+28%)
    Geographic Performance: North America (40%), Europe (35%), Asia-Pacific (25%)
    Customer Satisfaction: 87% (up from 82% in Q2)
    Key Challenges: Supply chain delays in Asia, increased competition in wireless audio market
    Recommendations: Expand marketing in Asia-Pacific, diversify suppliers, invest in R&D for next-gen audio products
    """
    
    messages = [
        SystemMessage(content="""You are a business analyst. Analyze the provided document and extract 
                      key insights, trends, and actionable recommendations for executive decision-making."""),
        HumanMessage(content=f"Please analyze this business report and provide executive summary with key insights:\n\n{sample_report}")
    ]
    
    response = llm.invoke(messages)
    print(f"AI Analysis: {response.content}")
    print("Business Value: Faster document processing, consistent analysis, key insight extraction")

def demonstrate_email_automation():
    """Demo 4: Professional Email Generation"""
    print("\n=== Demo 4: Professional Email Automation ===")
    print("Business Need: Generate professional emails for various business scenarios")
    
    messages = [
        SystemMessage(content="""You are a professional business communication specialist. 
                      Write clear, professional emails appropriate for corporate environments. 
                      Maintain professional tone while being concise and actionable."""),
        HumanMessage(content="""Write a follow-up email to a potential client who attended our 
                     product demo last week. We demonstrated our CRM software solution. 
                     The client seemed interested but hasn't responded yet. Include next steps.""")
    ]
    
    response = llm.invoke(messages)
    print(f"AI Generated Email: {response.content}")
    print("Business Value: Consistent professional communication, time savings, improved follow-up rates")

def demonstrate_product_development():
    """Demo 5: Product Feature Analysis"""
    print("\n=== Demo 5: Product Development Support ===")
    print("Business Need: Analyze market trends and suggest product improvements")
    
    messages = [
        SystemMessage(content="""You are a product strategist with expertise in tech products. 
                      Analyze market trends and provide strategic product recommendations 
                      based on customer needs and competitive landscape."""),
        HumanMessage(content="""Our mobile app currently has basic task management features. 
                     Competitors are adding AI-powered features like smart scheduling and 
                     automated priority setting. What new features should we consider 
                     to stay competitive?""")
    ]
    
    response = llm.invoke(messages)
    print(f"AI Recommendations: {response.content}")
    print("Business Value: Strategic insights, competitive analysis, innovation guidance")

def main():
    """Run all business requirement demonstrations"""
    print("GENERATIVE AI FOR BUSINESS APPLICATIONS")
    print("=" * 50)
    print("Demonstrating real-world business use cases for AI implementation")
    
    try:
        # Run all demonstrations
        demonstrate_customer_service()
        demonstrate_content_marketing()
        demonstrate_document_analysis()
        demonstrate_email_automation()
        demonstrate_product_development()
        
        print("\n" + "=" * 50)
        print("BUSINESS IMPACT SUMMARY:")
        print("✓ Automated customer service responses")
        print("✓ Rapid content creation and marketing copy")
        print("✓ Intelligent document analysis and insights")
        print("✓ Professional email automation")
        print("✓ Strategic product development guidance")
        print("\nROI Potential: Reduced manual work, faster decision-making, improved customer satisfaction")
        
    except Exception as e:
        print(f"Error in demonstration: {e}")
        print("Please check your Azure OpenAI configuration and API keys.")

if __name__ == "__main__":
    main()

