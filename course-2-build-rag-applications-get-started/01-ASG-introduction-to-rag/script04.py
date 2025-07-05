"""
Company Policies Q&A RAG Application Implementation
================================================================================

Module: script04.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module implements a RAG application for company policies Q&A system.
    It uses vector-based document retrieval and generation to answer employee
    questions about company policies including HR policies, IT policies, and
    finance policies with up-to-date information tracking.
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
    """Create sample company policy documents for indexing"""
    documents = [
        Document(
            page_content="Remote Work Policy: Employees are eligible for remote work arrangements with manager approval. Remote workers must maintain reliable internet connection, dedicated workspace, and be available during core business hours (9 AM - 3 PM local time). All remote work arrangements must be reviewed quarterly and documented in employee records.",
            metadata={"source": "hr_policies", "topic": "remote_work", "last_updated": "2024-01-15"}
        ),
        Document(
            page_content="Annual Leave Policy: Full-time employees accrue 2.5 days of annual leave per month, capped at 30 days annually. Leave requests must be submitted at least 2 weeks in advance for approval. Unused leave up to 5 days can be carried over to the next year. Leave encashment is allowed only upon resignation or retirement.",
            metadata={"source": "hr_policies", "topic": "annual_leave", "last_updated": "2024-02-01"}
        ),
        Document(
            page_content="Code of Conduct: All employees must maintain professional behavior, respect diversity, and avoid conflicts of interest. Harassment, discrimination, or inappropriate behavior will result in disciplinary action. Employees must report any violations through the anonymous reporting system or to HR directly.",
            metadata={"source": "hr_policies", "topic": "code_of_conduct", "last_updated": "2024-01-10"}
        ),
        Document(
            page_content="Data Security Policy: All company data must be handled according to classification levels (Public, Internal, Confidential, Restricted). Employees must use strong passwords, enable two-factor authentication, and never share login credentials. Data breaches must be reported immediately to the IT Security team.",
            metadata={"source": "it_policies", "topic": "data_security", "last_updated": "2024-03-01"}
        ),
        Document(
            page_content="Performance Review Policy: Annual performance reviews are conducted for all employees in Q4. Reviews include goal assessment, competency evaluation, and development planning. Mid-year check-ins are required to track progress. Performance ratings directly impact salary adjustments and promotion decisions.",
            metadata={"source": "hr_policies", "topic": "performance_review", "last_updated": "2024-01-20"}
        ),
        Document(
            page_content="Expense Reimbursement Policy: Business expenses must be pre-approved for amounts exceeding $500. All expenses require valid receipts and must be submitted within 30 days. Reimbursable expenses include travel, training, client entertainment, and necessary business supplies. Personal expenses are not reimbursable.",
            metadata={"source": "finance_policies", "topic": "expense_reimbursement", "last_updated": "2024-02-15"}
        ),
        Document(
            page_content="Sick Leave Policy: Employees are entitled to 12 days of sick leave annually. Medical certificates are required for absences exceeding 3 consecutive days. Sick leave can be used for personal illness, medical appointments, or caring for immediate family members. Unused sick leave does not carry over to the next year.",
            metadata={"source": "hr_policies", "topic": "sick_leave", "last_updated": "2024-01-25"}
        ),
        Document(
            page_content="Professional Development Policy: The company supports employee growth through training budgets up to $2,000 per employee annually. Training requests must align with job responsibilities and career development goals. Employees must complete training within 12 months and share learnings with their team.",
            metadata={"source": "hr_policies", "topic": "professional_development", "last_updated": "2024-02-10"}
        ),
        Document(
            page_content="Work From Home Equipment Policy: Company provides necessary equipment for remote work including laptop, monitor, and office chair. Employees are responsible for equipment maintenance and security. Equipment must be returned upon resignation or role change. Personal use of company equipment is permitted within reasonable limits.",
            metadata={"source": "it_policies", "topic": "equipment_policy", "last_updated": "2024-02-20"}
        ),
        Document(
            page_content="Flexible Working Hours Policy: Core hours are 9 AM to 3 PM when all team members must be available. Employees can start work between 7 AM and 10 AM, completing 8 hours daily. Schedule changes must be approved by direct manager and communicated to the team. Consistent schedules are preferred for team coordination.",
            metadata={"source": "hr_policies", "topic": "flexible_hours", "last_updated": "2024-03-05"}
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
    You are a helpful HR assistant specializing in company policies. Answer the employee's question based on the provided company policy documents.
    
    Policy Context:
    {context}
    
    Employee Question: {question}
    
    Please provide a clear and comprehensive answer based on the company policies provided. If the policies don't contain enough information to fully answer the question, mention that and suggest contacting HR for clarification. Always reference the relevant policy when possible and include the last updated date if available.
    
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
    with open("rag_graph_04.png", "wb") as f:
        f.write(image)
    
    return workflow

def initialize_vector_store():
    """Initialize vector store with company policy documents (done once)"""
    global vector_store
    
    print("\n=== Initializing Vector Store with Company Policies ===")
    
    # Create a temporary directory for Chroma
    persist_directory = tempfile.mkdtemp()
    
    try:
        # Initialize Chroma vector store
        vector_store = Chroma(
            collection_name="company_policies_collection",
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        # Create company policy documents
        documents = create_sample_documents()
        
        # Generate unique IDs for documents
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        # Add documents to vector store
        vector_store.add_documents(documents=documents, ids=uuids)
        
        print(f"Successfully indexed {len(documents)} company policy documents")
        return True
        
    except Exception as e:
        print(f"Error during policy indexing: {e}")
        return False

def run_rag_demo():
    """Run the RAG demonstration"""
    print("\n" + "="*50)
    print("Company Policies Q&A RAG Application")
    print("="*50)
    
    # Initialize vector store once before creating the graph
    if not initialize_vector_store():
        print("Failed to initialize vector store. Exiting.")
        return
    
    # Create RAG graph
    rag_app = create_rag_graph()
    
    # Demo queries
    queries = [
        "What is the remote work policy?",
        "How many days of annual leave do I get?",
        "What should I do if I witness harassment at work?",
        "What are the data security requirements?",
        "When are performance reviews conducted?",
        "What expenses can I get reimbursed for?",
        "How many sick days am I entitled to?",
        "What is the professional development budget?",
        "Can I work flexible hours?",
        "What happens if I don't use all my vacation days?"
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
    
    print("\nCompany Policies Q&A Demo completed!")

if __name__ == "__main__":
    run_rag_demo()

