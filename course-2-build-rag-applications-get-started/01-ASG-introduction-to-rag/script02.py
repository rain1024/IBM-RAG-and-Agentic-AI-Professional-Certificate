import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from typing import List, Dict, Any, TypedDict
import json

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

# Initialize Tavily Search Tool
tavily_search = TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=5,
    topic="general",
)

class RAGState(TypedDict):
    """State for the RAG application"""
    query: str
    analyzed_query: str
    search_keywords: List[str]
    documents: List[Document]
    context: str
    response: str

def analyze_query(state: RAGState) -> RAGState:
    """Analyze the user query to extract key information and search keywords"""
    print(f"\n=== Analyzing Query: '{state['query']}' ===")
    
    analyze_prompt = ChatPromptTemplate.from_template("""
    You are a query analyzer. Analyze the user's question and extract:
    1. Key search keywords/phrases that would help find relevant information
    2. A refined version of the query that's optimized for search
    
    Original Question: {query}
    
    Please respond in JSON format with the following structure:
    {{
        "analyzed_query": "refined search-optimized version of the query",
        "search_keywords": ["keyword1", "keyword2", "keyword3"]
    }}
    
    Focus on extracting the most important terms and concepts from the question.
    """)
    
    try:
        analyze_chain = (
            analyze_prompt
            | llm
            | StrOutputParser()
        )
        
        analysis_result = analyze_chain.invoke({
            "query": state["query"]
        })
        
        # Parse the JSON response
        try:
            analysis_data = json.loads(analysis_result)
            analyzed_query = analysis_data.get("analyzed_query", state["query"])
            search_keywords = analysis_data.get("search_keywords", [])
        except json.JSONDecodeError:
            print("Failed to parse analysis result, using original query")
            analyzed_query = state["query"]
            search_keywords = []
        
        print(f"Analyzed Query: {analyzed_query}")
        print(f"Search Keywords: {search_keywords}")
        
        return {
            **state, 
            "analyzed_query": analyzed_query,
            "search_keywords": search_keywords
        }
        
    except Exception as e:
        print(f"Error during query analysis: {e}")
        return {
            **state, 
            "analyzed_query": state["query"],
            "search_keywords": []
        }

def search_documents(state: RAGState) -> RAGState:
    """Search for relevant documents using Tavily Search"""
    print(f"\n=== Searching with Tavily for: '{state['analyzed_query']}' ===")
    

    try:
        # Use the analyzed query for search
        search_query = state["analyzed_query"]
        
        # Perform Tavily search
        search_results = tavily_search.invoke({"query": search_query})
        
        if not search_results:
            print("No search results found")
            return state
        
        # Parse search results
        if isinstance(search_results, str):
            try:
                search_data = json.loads(search_results)
                results = search_data.get("results", [])
            except json.JSONDecodeError:
                print("Failed to parse search results")
                return state
        else:
            results = search_results.get("results", [])
        
        print(f"Found {len(results)} search results")
        
        # Create documents from search results
        documents = []
        context_parts = []
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "")
            
            if content:
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": url,
                        "title": title,
                        "search_rank": i
                    }
                )
                documents.append(doc)
                context_parts.append(f"Document {i} - {title} (Source: {url}):\n{content}")
                
                print(f"Retrieved: {title}")
        
        context = "\n\n".join(context_parts)
        
        print(f"Successfully retrieved content from {len(documents)} sources")
        
        return {**state, "documents": documents, "context": context}
        
    except Exception as e:
        print(f"Error during Tavily search: {e}")
        return state

def generate_response(state: RAGState) -> RAGState:
    """Generate response using LLM with retrieved context"""
    print(f"\n=== Generating Response ===")
    
    if not state["context"]:
        print("No context available. Please search for documents first.")
        return state
    
    # Create RAG prompt template
    rag_prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant. Answer the user's question based on the provided context from web search results.
    
    Context:
    {context}
    
    Original Question: {question}
    
    Please provide a comprehensive and accurate answer based on the context provided. 
    - Use specific information from the sources when possible
    - If the context doesn't contain enough information to fully answer the question, mention that clearly
    - Cite relevant sources when appropriate
    - Provide a well-structured and informative response
    
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
    """Create LangGraph workflow for RAG with three steps"""
    graph = StateGraph(RAGState)
    
    # Add nodes for the three steps
    graph.add_node("analyze", analyze_query)
    graph.add_node("search", search_documents)
    graph.add_node("generate", generate_response)
    
    # Add edges to create the workflow: analyze -> search -> generate
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "search")
    graph.add_edge("search", "generate")
    graph.add_edge("generate", END)
    
    workflow = graph.compile()
    
    # Save graph visualization
    try:
        image = workflow.get_graph().draw_mermaid_png()
        with open("rag_graph_02.png", "wb") as f:
            f.write(image)
        print("Graph visualization saved as rag_graph.png")
    except Exception as e:
        print(f"Could not save graph visualization: {e}")
    
    return workflow

def run_rag_demo():
    """Run the RAG demonstration"""
    print("\n" + "="*50)
    print("RAG Application with Tavily Search")
    print("Three-Step Process: Analyze -> Search -> Generate")
    print("="*50)
    
    # Create RAG graph
    rag_app = create_rag_graph()
    
    # Demo queries
    queries = [
        "What is the planet with the most moons?",
        "Who is the current Prime Minister of Vietnam?",
        "What is the best-selling smartphone in the world?",
        "Which company has the highest market capitalization?",
        "What is the current price of Bitcoin?",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Processing Query: {query}")
        print(f"{'='*60}")
        
        # Initialize state for this query
        state = RAGState(
            query=query,
            analyzed_query="",
            search_keywords=[],
            documents=[],
            context="",
            response=""
        )
        
        # Run RAG workflow
        try:
            final_state = rag_app.invoke(state)
            
            print(f"\n--- Final Results ---")
            print(f"Original Query: {final_state['query']}")
            print(f"Analyzed Query: {final_state['analyzed_query']}")
            print(f"Search Keywords: {final_state['search_keywords']}")
            print(f"Response: {final_state['response']}")
            
            if final_state["documents"]:
                print(f"\n--- Retrieved Documents ({len(final_state['documents'])} sources) ---")
                for i, doc in enumerate(final_state["documents"], 1):
                    print(f"{i}. {doc.metadata.get('title', 'Untitled')}")
                    print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
                    print(f"   Content: {doc.page_content[:150]}...")
                    print()
            
        except Exception as e:
            print(f"Error processing query: {e}")
        
        print("\n" + "-"*60)
    
    print("\nRAG Demo completed!")

if __name__ == "__main__":
    run_rag_demo()

