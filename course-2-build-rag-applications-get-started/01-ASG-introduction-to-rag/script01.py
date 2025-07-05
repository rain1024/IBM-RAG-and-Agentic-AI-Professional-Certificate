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
    """Create sample documents for indexing"""
    documents = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions.",
            metadata={"source": "ai_basics", "topic": "machine_learning"}
        ),
        Document(
            page_content="Deep learning is a specialized form of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It's particularly effective for tasks like image recognition, natural language processing, and speech recognition.",
            metadata={"source": "ai_basics", "topic": "deep_learning"}
        ),
        Document(
            page_content="Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in a valuable way.",
            metadata={"source": "ai_basics", "topic": "nlp"}
        ),
        Document(
            page_content="Computer vision is an area of AI that enables computers to interpret and understand visual information from the world. It involves techniques for acquiring, processing, analyzing, and understanding digital images and videos.",
            metadata={"source": "ai_basics", "topic": "computer_vision"}
        ),
        Document(
            page_content="Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties for its actions and learns to maximize cumulative rewards over time.",
            metadata={"source": "ai_basics", "topic": "reinforcement_learning"}
        ),
        Document(
            page_content="Supervised learning is a machine learning approach where models are trained on labeled data. The algorithm learns from input-output pairs to make predictions on new, unseen data.",
            metadata={"source": "ai_basics", "topic": "supervised_learning"}
        ),
        Document(
            page_content="Unsupervised learning is a type of machine learning that finds patterns in data without labeled examples. Common techniques include clustering, dimensionality reduction, and association rule learning.",
            metadata={"source": "ai_basics", "topic": "unsupervised_learning"}
        ),
        Document(
            page_content="Artificial General Intelligence (AGI) refers to a form of AI that possesses the ability to understand, learn, and apply knowledge across a wide range of tasks at a level comparable to human intelligence.",
            metadata={"source": "ai_basics", "topic": "agi"}
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
    You are a helpful AI assistant. Answer the user's question based on the provided context.
    
    Context:
    {context}
    
    Question: {question}
    
    Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, mention that and provide what information you can.
    
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
    with open("rag_graph_01.png", "wb") as f:
        f.write(image)
    
    return workflow

def initialize_vector_store():
    """Initialize vector store with sample documents (done once)"""
    global vector_store
    
    print("\n=== Initializing Vector Store ===")
    
    # Create a temporary directory for Chroma
    persist_directory = tempfile.mkdtemp()
    
    try:
        # Initialize Chroma vector store
        vector_store = Chroma(
            collection_name="rag_collection",
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        # Create sample documents
        documents = create_sample_documents()
        
        # Generate unique IDs for documents
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        # Add documents to vector store
        vector_store.add_documents(documents=documents, ids=uuids)
        
        print(f"Successfully indexed {len(documents)} documents")
        return True
        
    except Exception as e:
        print(f"Error during indexing: {e}")
        return False

def run_rag_demo():
    """Run the RAG demonstration"""
    print("\n" + "="*50)
    print("RAG Application with Chroma Vector Store")
    print("="*50)
    
    # Initialize vector store once before creating the graph
    if not initialize_vector_store():
        print("Failed to initialize vector store. Exiting.")
        return
    
    # Create RAG graph
    rag_app = create_rag_graph()
    
    # Demo queries
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "What is the difference between supervised and unsupervised learning?",
        "Explain reinforcement learning",
        "What is computer vision used for?",
        "What is Google?"
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
    
    print("\nRAG Demo completed!")

if __name__ == "__main__":
    run_rag_demo()

