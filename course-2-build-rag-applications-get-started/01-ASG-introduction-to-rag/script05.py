"""
Mobile Company Policies Q&A RAG Application Implementation
================================================================================

Module: script05.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module implements a RAG application for mobile company policies Q&A system.
    It uses vector-based document retrieval and generation to answer customer
    questions about mobile company policies including billing policies, data plans,
    roaming policies, and customer service policies with up-to-date information tracking.
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from typing import List, Dict, Any, TypedDict
import tempfile
import shutil
from uuid import uuid4

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

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002"),
)

# Global vector store
vector_store = None

class RAGState(TypedDict):
    """State for the RAG application"""
    query: str
    documents: List[Document]
    context: str
    response: str
    indexed: bool

def create_sample_documents() -> List[Document]:
    """Create sample mobile company policy documents for indexing"""
    documents = [
        Document(
            page_content="Data Plan Policy: All data plans include unlimited talk and text within Vietnam. Data speeds may be reduced after reaching monthly limit. Fair usage policy applies to unlimited plans with 50GB high-speed data threshold. Customers can purchase additional data packages through the mobile app or customer service.",
            metadata={"source": "service_policies", "topic": "data_plans", "last_updated": "2024-03-15"}
        ),
        Document(
            page_content="Billing Policy: Monthly bills are generated on the same date each month. Payment is due within 30 days of bill generation. Late payment fees of 50,000 VND apply after 30 days. Customers can set up auto-payment, pay online, or visit retail stores. Billing disputes must be reported within 90 days.",
            metadata={"source": "billing_policies", "topic": "billing_payment", "last_updated": "2024-03-10"}
        ),
        Document(
            page_content="International Roaming Policy: Roaming services must be activated before traveling abroad. Daily roaming packages available for data and calls. Roaming rates vary by country and zone. Customers receive SMS notifications about roaming charges. Roaming can be disabled to prevent unexpected charges.",
            metadata={"source": "roaming_policies", "topic": "international_roaming", "last_updated": "2024-03-01"}
        ),
        Document(
            page_content="Contract Termination Policy: Customers can terminate contracts with 30 days written notice. Early termination fees apply for contracts under 24 months. Device payment plans continue until fully paid. Final bill includes all charges up to termination date. Number portability available within 30 days of termination.",
            metadata={"source": "contract_policies", "topic": "contract_termination", "last_updated": "2024-02-28"}
        ),
        Document(
            page_content="Customer Support Policy: 24/7 customer support available via hotline 18001090. Online chat support available 6 AM to 10 PM daily. Technical support includes network troubleshooting and device assistance. Complaints escalated to management within 48 hours. Customer satisfaction surveys sent after each interaction.",
            metadata={"source": "support_policies", "topic": "customer_support", "last_updated": "2024-03-05"}
        ),
        Document(
            page_content="Device Insurance Policy: Optional device insurance covers accidental damage, theft, and water damage. Monthly premium based on device value. Claims require police report for theft. Maximum 2 claims per year. Replacement devices may be refurbished units of same model or equivalent.",
            metadata={"source": "insurance_policies", "topic": "device_insurance", "last_updated": "2024-02-20"}
        ),
        Document(
            page_content="Network Coverage Policy: 4G coverage available in 95% of populated areas. 5G coverage expanding in major cities. Network maintenance may cause temporary service interruptions. Customers notified 24 hours in advance for planned maintenance. Service credits available for extended outages exceeding 4 hours.",
            metadata={"source": "network_policies", "topic": "network_coverage", "last_updated": "2024-03-12"}
        ),
        Document(
            page_content="Privacy Policy: Customer data protected according to Vietnam Personal Data Protection regulations. Data used for service delivery and improvement. Marketing communications require customer consent. Customers can request data deletion or modification. Third-party data sharing limited to legal requirements only.",
            metadata={"source": "privacy_policies", "topic": "data_privacy", "last_updated": "2024-03-08"}
        ),
        Document(
            page_content="Upgrade Policy: Device upgrades available after 12 months with eligible plans. Trade-in values applied toward new device cost. Customers must complete device payments before upgrading. Plan changes may be required for new devices. Upgrade eligibility checked through customer portal or retail stores.",
            metadata={"source": "upgrade_policies", "topic": "device_upgrade", "last_updated": "2024-02-15"}
        ),
        Document(
            page_content="Fair Usage Policy: Unlimited plans subject to fair usage limits. Heavy users may experience reduced speeds during peak hours. Tethering limited to 20GB per month on unlimited plans. Commercial use of consumer plans prohibited. Network management ensures quality service for all customers.",
            metadata={"source": "usage_policies", "topic": "fair_usage", "last_updated": "2024-03-18"}
        ),
        Document(
            page_content="Refund Policy: New customers have 14-day satisfaction guarantee. Full refund available for service cancellation within 14 days. Device returns must be in original condition. Restocking fee applies to opened devices. Refunds processed within 7-10 business days to original payment method.",
            metadata={"source": "refund_policies", "topic": "refund_guarantee", "last_updated": "2024-03-02"}
        ),
        Document(
            page_content="SIM Card Policy: Physical SIM cards available at retail stores for 50,000 VND. eSIM activation available for compatible devices. SIM replacement fee applies for lost or damaged cards. Number transfer between SIM types supported. SIM cards must be activated within 30 days of purchase.",
            metadata={"source": "sim_policies", "topic": "sim_management", "last_updated": "2024-03-07"}
        )
    ]
    return documents



def retrieve_documents(state: RAGState) -> RAGState:
    """Retrieve relevant documents based on query"""
    global vector_store
    
    print(f"\n=== Retrieving Documents for Query: '{state['query']}' ===")
    
    if not vector_store:
        print("Vector store not initialized. Please index documents first.")
        return state
    
    try:
        # Perform similarity search
        results = vector_store.similarity_search(
            state["query"],
            k=3  # Retrieve top 3 most relevant documents
        )
        
        # Create context from retrieved documents
        context_parts = []
        for i, doc in enumerate(results, 1):
            context_parts.append(f"Document {i}: {doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        print(f"Retrieved {len(results)} relevant documents")
        
        return {**state, "documents": results, "context": context}
        
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return state

def generate_response(state: RAGState) -> RAGState:
    """Generate response using LLM with retrieved context"""
    print(f"\n=== Generating Response ===")
    
    if not state["context"]:
        print("No context available. Please retrieve documents first.")
        return state
    
    # Create RAG prompt template
    rag_prompt = ChatPromptTemplate.from_template("""
    You are a helpful customer service assistant specializing in mobile company policies. Answer the customer's question based on the provided mobile company policy documents.
    
    Policy Context:
    {context}
    
    Customer Question: {question}
    
    Please provide a clear and comprehensive answer based on the mobile company policies provided. If the policies don't contain enough information to fully answer the question, mention that and suggest contacting customer service at 18001090 for clarification. Always reference the relevant policy when possible and include the last updated date if available. Be friendly and professional in your response.
    
    Answer:
    """)
    
    try:
        # Create the RAG chain
        rag_chain = (
            rag_prompt
            | llm
            | StrOutputParser()
        )
        
        # Generate response
        # print("State: ", state)
        response = rag_chain.invoke({
            "context": state["context"],
            "question": state["query"]
        })
        
        print("Response generated successfully")
        
        return {**state, "response": response}
        
    except Exception as e:
        print(f"Error during response generation: {e}")
        return state

def create_rag_graph():
    """Create LangGraph workflow for RAG"""
    graph = StateGraph(RAGState)
    
    # Add nodes
    graph.add_node("retrieve", retrieve_documents)
    graph.add_node("generate", generate_response)
    
    # Add edges
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    
    workflow = graph.compile()
    image = workflow.get_graph().draw_mermaid_png()
    with open("rag_graph_05.png", "wb") as f:
        f.write(image)
    
    return workflow

def initialize_vector_store():
    """Initialize vector store with mobile company policy documents (done once)"""
    global vector_store
    
    print("\n=== Initializing Vector Store with Mobile Company Policies ===")
    
    # Create a temporary directory for Chroma
    persist_directory = tempfile.mkdtemp()
    
    try:
        # Initialize Chroma vector store
        vector_store = Chroma(
            collection_name="mobile_company_policies_collection",
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        # Create mobile company policy documents
        documents = create_sample_documents()
        
        # Generate unique IDs for documents
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        # Add documents to vector store
        vector_store.add_documents(documents=documents, ids=uuids)
        
        print(f"Successfully indexed {len(documents)} mobile company policy documents")
        return True
        
    except Exception as e:
        print(f"Error during policy indexing: {e}")
        return False

def run_rag_demo():
    """Run the RAG demonstration"""
    print("\n" + "="*50)
    print("Mobile Company Policies Q&A RAG Application")
    print("="*50)
    
    # Initialize vector store once before creating the graph
    if not initialize_vector_store():
        print("Failed to initialize vector store. Exiting.")
        return
    
    # Create RAG graph
    rag_app = create_rag_graph()
    
    # Demo queries
    queries = [
        "What happens if I exceed my data limit?",
        "How do I pay my monthly bill?",
        "Can I use my phone abroad?",
        "How do I cancel my contract?",
        "What is your customer support number?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Processing Query: {query}")
        print(f"{'='*60}")
        
        # Initialize state for this query
        state = RAGState(
            query=query,
            documents=[],
            context="",
            response="",
            indexed=True  # Already indexed
        )
        
        # Run RAG workflow
        try:
            final_state = rag_app.invoke(state)
            
            print(f"\n--- Final Response ---")
            print(f"Query: {final_state['query']}")
            print(f"Response: {final_state['response']}")
            
            if final_state["documents"]:
                print(f"\n--- Retrieved Documents ---")
                for i, doc in enumerate(final_state["documents"], 1):
                    print(f"{i}. {doc.page_content[:100]}...")
                    print(f"   Metadata: {doc.metadata}")
            
        except Exception as e:
            print(f"Error processing query: {e}")
        
        print("\n" + "-"*60)
    
    print("\nMobile Company Policies Q&A Demo completed!")

if __name__ == "__main__":
    run_rag_demo()

