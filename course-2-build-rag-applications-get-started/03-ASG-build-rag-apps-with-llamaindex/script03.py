"""
Company Policies Q&A RAG Application Implementation with LlamaIndex
================================================================================

Module: script01.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module implements a RAG application for company policies Q&A system.
    It uses LlamaIndex for vector-based document retrieval and generation to answer employee
    questions about company policies including HR policies, IT policies, and
    finance policies with up-to-date information tracking.
"""

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureOpenAI(
    model="gpt-4",
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    temperature=0.7,
    max_tokens=1000
)

# Initialize embeddings
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
)

# Configure LlamaIndex settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

class CompanyPoliciesRAG:
    """Company Policies RAG Application using LlamaIndex"""
    
    def __init__(self):
        self.index = None
        self.query_engine = None
        self.documents = []
        
    def create_sample_documents(self) -> List[Document]:
        """Create sample company policy documents for indexing"""
        documents = [
            Document(
                text="Remote Work Policy: Employees are eligible for remote work arrangements with manager approval. Remote workers must maintain reliable internet connection, dedicated workspace, and be available during core business hours (9 AM - 3 PM local time). All remote work arrangements must be reviewed quarterly and documented in employee records.",
                metadata={"source": "hr_policies", "topic": "remote_work", "last_updated": "2024-01-15"}
            ),
            Document(
                text="Annual Leave Policy: Full-time employees accrue 2.5 days of annual leave per month, capped at 30 days annually. Leave requests must be submitted at least 2 weeks in advance for approval. Unused leave up to 5 days can be carried over to the next year. Leave encashment is allowed only upon resignation or retirement.",
                metadata={"source": "hr_policies", "topic": "annual_leave", "last_updated": "2024-02-01"}
            ),
            Document(
                text="Code of Conduct: All employees must maintain professional behavior, respect diversity, and avoid conflicts of interest. Harassment, discrimination, or inappropriate behavior will result in disciplinary action. Employees must report any violations through the anonymous reporting system or to HR directly.",
                metadata={"source": "hr_policies", "topic": "code_of_conduct", "last_updated": "2024-01-10"}
            ),
            Document(
                text="Data Security Policy: All company data must be handled according to classification levels (Public, Internal, Confidential, Restricted). Employees must use strong passwords, enable two-factor authentication, and never share login credentials. Data breaches must be reported immediately to the IT Security team.",
                metadata={"source": "it_policies", "topic": "data_security", "last_updated": "2024-03-01"}
            ),
            Document(
                text="Performance Review Policy: Annual performance reviews are conducted for all employees in Q4. Reviews include goal assessment, competency evaluation, and development planning. Mid-year check-ins are required to track progress. Performance ratings directly impact salary adjustments and promotion decisions.",
                metadata={"source": "hr_policies", "topic": "performance_review", "last_updated": "2024-01-20"}
            ),
            Document(
                text="Expense Reimbursement Policy: Business expenses must be pre-approved for amounts exceeding $500. All expenses require valid receipts and must be submitted within 30 days. Reimbursable expenses include travel, training, client entertainment, and necessary business supplies. Personal expenses are not reimbursable.",
                metadata={"source": "finance_policies", "topic": "expense_reimbursement", "last_updated": "2024-02-15"}
            ),
            Document(
                text="Sick Leave Policy: Employees are entitled to 12 days of sick leave annually. Medical certificates are required for absences exceeding 3 consecutive days. Sick leave can be used for personal illness, medical appointments, or caring for immediate family members. Unused sick leave does not carry over to the next year.",
                metadata={"source": "hr_policies", "topic": "sick_leave", "last_updated": "2024-01-25"}
            ),
            Document(
                text="Professional Development Policy: The company supports employee growth through training budgets up to $2,000 per employee annually. Training requests must align with job responsibilities and career development goals. Employees must complete training within 12 months and share learnings with their team.",
                metadata={"source": "hr_policies", "topic": "professional_development", "last_updated": "2024-02-10"}
            ),
            Document(
                text="Work From Home Equipment Policy: Company provides necessary equipment for remote work including laptop, monitor, and office chair. Employees are responsible for equipment maintenance and security. Equipment must be returned upon resignation or role change. Personal use of company equipment is permitted within reasonable limits.",
                metadata={"source": "it_policies", "topic": "equipment_policy", "last_updated": "2024-02-20"}
            ),
            Document(
                text="Flexible Working Hours Policy: Core hours are 9 AM to 3 PM when all team members must be available. Employees can start work between 7 AM and 10 AM, completing 8 hours daily. Schedule changes must be approved by direct manager and communicated to the team. Consistent schedules are preferred for team coordination.",
                metadata={"source": "hr_policies", "topic": "flexible_hours", "last_updated": "2024-03-05"}
            )
        ]
        return documents
    
    def initialize_index(self):
        """Initialize the vector store index with company policy documents"""
        print("\n=== Initializing LlamaIndex with Company Policies ===")
        
        try:
            # Create company policy documents
            self.documents = self.create_sample_documents()
            
            # Create vector store index
            self.index = VectorStoreIndex.from_documents(
                self.documents,
                show_progress=True
            )
            
            print(f"Successfully indexed {len(self.documents)} company policy documents")
            return True
            
        except Exception as e:
            print(f"Error during policy indexing: {e}")
            return False
    
    def setup_query_engine(self):
        """Setup the query engine with custom RAG prompt"""
        if not self.index:
            print("Index not initialized. Please initialize index first.")
            return
        
        # Custom RAG prompt template
        rag_prompt_template = PromptTemplate(
            """
            You are a helpful HR assistant specializing in company policies. Answer the employee's question based on the provided company policy documents.
            
            Context information is below:
            ---------------------
            {context_str}
            ---------------------
            
            Given the context information and not prior knowledge, answer the employee's question.
            Please provide a clear and comprehensive answer based on the company policies provided. 
            If the policies don't contain enough information to fully answer the question, mention that and suggest contacting HR for clarification. 
            Always reference the relevant policy when possible and include the last updated date if available.
            
            Employee Question: {query_str}
            
            Answer:
            """
        )
        
        try:
            # Create query engine with custom prompt
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=3,  # Retrieve top 3 most relevant documents
                text_qa_template=rag_prompt_template,
                response_mode="compact"
            )
            
            print("Query engine setup completed successfully")
            
        except Exception as e:
            print(f"Error setting up query engine: {e}")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        if not self.query_engine:
            return {"error": "Query engine not initialized"}
        
        print(f"\n=== Processing Query: '{question}' ===")
        
        try:
            # Query the index
            response = self.query_engine.query(question)
            
            # Extract source documents
            source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
            
            result = {
                "query": question,
                "response": str(response),
                "source_documents": []
            }
            
            # Add source document information
            for i, node in enumerate(source_nodes, 1):
                result["source_documents"].append({
                    "rank": i,
                    "content": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    "metadata": node.metadata,
                    "score": node.score if hasattr(node, 'score') else None
                })
            
            print(f"Retrieved {len(source_nodes)} relevant documents")
            return result
            
        except Exception as e:
            print(f"Error during query: {e}")
            return {"error": str(e)}

def run_rag_demo():
    """Run the RAG demonstration"""
    print("\n" + "="*50)
    print("Company Policies Q&A RAG Application with LlamaIndex")
    print("="*50)
    
    # Initialize RAG application
    rag_app = CompanyPoliciesRAG()
    
    # Initialize index
    if not rag_app.initialize_index():
        print("Failed to initialize index. Exiting.")
        return
    
    # Setup query engine
    rag_app.setup_query_engine()
    
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
        
        # Query the RAG system
        result = rag_app.query(query)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            continue
        
        print(f"\n--- Final Response ---")
        print(f"Query: {result['query']}")
        print(f"Response: {result['response']}")
        
        if result["source_documents"]:
            print(f"\n--- Retrieved Documents ---")
            for doc in result["source_documents"]:
                print(f"{doc['rank']}. {doc['content']}")
                print(f"   Metadata: {doc['metadata']}")
                if doc['score']:
                    print(f"   Relevance Score: {doc['score']:.4f}")
        
        print("\n" + "-"*60)
    
    print("\nCompany Policies Q&A Demo completed!")

if __name__ == "__main__":
    run_rag_demo()