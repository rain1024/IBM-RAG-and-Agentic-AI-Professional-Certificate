"""
Response Synthesizers Demo: LlamaIndex Prompt Synthesis Strategies
================================================================================

Module: script05.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates various response synthesizers in LlamaIndex.
    It shows how different synthesizers combine information from multiple sources
    to generate comprehensive responses, including TreeSummarize, Accumulate,
    CompactAndRefine, and SimpleResponseSynthesizer strategies.
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import textwrap
import time

# LlamaIndex imports
from llama_index.core.response_synthesizers import (
    TreeSummarize,
    Accumulate,
    CompactAndRefine,
    get_response_synthesizer
)
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.llms import LLM
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

# Load environment variables
load_dotenv()

class ResponseSynthesizerDemo:
    """Demo class for comparing different response synthesizers in LlamaIndex"""
    
    def __init__(self):
        self.sample_documents = self.create_sample_documents()
        self.sample_queries = self.create_sample_queries()
        self.llm = None
        self.embed_model = None
        self.index = None
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize LLM and embedding models"""
        try:
            self.llm = AzureOpenAI(
                model="gpt-4",
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
                temperature=0.1,
                max_tokens=1000,
            )
            
            self.embed_model = AzureOpenAIEmbedding(
                model="text-embedding-ada-002",
                deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
            )
            
            print("✓ Models initialized successfully")
            
        except Exception as e:
            print(f"⚠ Warning: Could not initialize models: {e}")
            print("Running in demo mode - will show structure without actual LLM calls")
            self.llm = None
            self.embed_model = None
    
    def create_sample_documents(self) -> List[Document]:
        """Create sample documents for demonstration"""
        documents = [
            Document(
                text="""
                Climate Change and Global Warming Overview
                
                Climate change refers to long-term shifts in global temperatures and weather patterns. 
                While climate variations are natural, scientific evidence shows that human activities, 
                particularly greenhouse gas emissions from burning fossil fuels, have been the primary 
                driver of climate change since the 1950s.
                
                The main greenhouse gases include carbon dioxide (CO2), methane (CH4), nitrous oxide (N2O), 
                and fluorinated gases. CO2 is the most significant contributor, accounting for about 76% 
                of total greenhouse gas emissions. The concentration of CO2 in the atmosphere has increased 
                by over 40% since pre-industrial times.
                
                Key indicators of climate change include rising global temperatures, melting ice caps, 
                rising sea levels, and changing precipitation patterns. The global average temperature 
                has risen by approximately 1.1°C since the late 1800s.
                """,
                metadata={"source": "climate_overview", "category": "environmental"}
            ),
            Document(
                text="""
                Impacts of Climate Change on Ecosystems
                
                Climate change has profound effects on ecosystems worldwide. Rising temperatures affect 
                the timing of biological events such as flowering, migration, and reproduction. Many 
                species are shifting their ranges toward cooler regions, typically poleward or to 
                higher elevations.
                
                Ocean acidification, caused by increased CO2 absorption, threatens marine ecosystems. 
                Coral reefs are particularly vulnerable, with widespread bleaching events occurring 
                more frequently. Arctic ecosystems face severe challenges as sea ice diminishes, 
                affecting polar bears, seals, and Arctic cod.
                
                Forest ecosystems are experiencing increased wildfire frequency and intensity. 
                Droughts stress vegetation, making forests more susceptible to insect infestations 
                and disease outbreaks. Some tree species are migrating northward, altering forest 
                composition and biodiversity.
                """,
                metadata={"source": "ecosystem_impacts", "category": "environmental"}
            ),
            Document(
                text="""
                Renewable Energy Solutions
                
                Renewable energy technologies offer promising solutions to reduce greenhouse gas 
                emissions. Solar photovoltaic (PV) technology has experienced dramatic cost reductions, 
                with prices falling by over 80% in the past decade. Wind energy has also become 
                increasingly competitive, with offshore wind showing particular promise.
                
                Hydroelectric power remains the largest source of renewable electricity globally, 
                providing about 16% of world electricity generation. However, new hydroelectric 
                projects face environmental and social concerns. Geothermal energy offers baseload 
                power generation in suitable locations.
                
                Energy storage technologies, particularly lithium-ion batteries, are crucial for 
                integrating intermittent renewable sources. Smart grid technologies help optimize 
                energy distribution and consumption. The transition to renewable energy requires 
                significant infrastructure investments but offers long-term economic and environmental benefits.
                """,
                metadata={"source": "renewable_energy", "category": "technology"}
            ),
            Document(
                text="""
                Climate Change Mitigation Strategies
                
                Effective climate change mitigation requires comprehensive strategies across multiple 
                sectors. Carbon pricing mechanisms, including carbon taxes and cap-and-trade systems, 
                provide economic incentives for emission reductions. Many countries have implemented 
                or are considering carbon pricing policies.
                
                Energy efficiency improvements in buildings, transportation, and industry can 
                significantly reduce emissions. Green building standards, electric vehicle adoption, 
                and industrial process optimization are key areas for improvement. Behavioral changes, 
                such as reducing energy consumption and changing dietary habits, also contribute 
                to mitigation efforts.
                
                Nature-based solutions, including reforestation, afforestation, and sustainable 
                agriculture practices, can sequester carbon while providing co-benefits for 
                biodiversity and ecosystem services. International cooperation through agreements 
                like the Paris Climate Agreement is essential for coordinated global action.
                """,
                metadata={"source": "mitigation_strategies", "category": "policy"}
            ),
            Document(
                text="""
                Climate Change Adaptation Measures
                
                Adaptation to climate change involves adjusting systems and societies to minimize 
                harm from climate impacts. Coastal communities are implementing sea-level rise 
                adaptation measures, including sea walls, managed retreat, and ecosystem-based 
                protection using mangroves and wetlands.
                
                Agricultural adaptation strategies include developing drought-resistant crops, 
                improving irrigation efficiency, and diversifying farming systems. Water management 
                adaptation involves improving storage capacity, reducing water waste, and developing 
                alternative water sources including desalination and water recycling.
                
                Urban planning increasingly incorporates climate resilience, including green 
                infrastructure, improved drainage systems, and heat island mitigation. Early 
                warning systems and disaster preparedness help communities respond to extreme 
                weather events. Health systems are adapting to address climate-related health 
                risks, including heat stress and vector-borne diseases.
                """,
                metadata={"source": "adaptation_measures", "category": "policy"}
            )
        ]
        
        return documents
    
    def create_sample_queries(self) -> List[str]:
        """Create sample queries for testing synthesizers"""
        return [
            "What are the main causes of climate change?",
            "How does climate change affect ecosystems and biodiversity?",
            "What renewable energy solutions are available to combat climate change?",
            "What are the key strategies for climate change mitigation?",
            "How can communities adapt to climate change impacts?",
            "What is the relationship between greenhouse gases and global warming?",
            "How do renewable energy technologies compare in terms of effectiveness?",
            "What are the economic implications of climate change mitigation and adaptation?"
        ]
    
    def setup_index(self):
        """Create vector index from sample documents"""
        if not self.embed_model:
            print("Embedding model not available, cannot create index")
            return False
        
        try:
            # Parse documents into nodes
            node_parser = SentenceSplitter(chunk_size=400, chunk_overlap=50)
            nodes = node_parser.get_nodes_from_documents(self.sample_documents)
            
            # Create vector index
            self.index = VectorStoreIndex(nodes, embed_model=self.embed_model)
            print(f"✓ Created index with {len(nodes)} nodes")
            return True
            
        except Exception as e:
            print(f"Error creating index: {e}")
            return False
    
    def demo_basic_synthesizers(self):
        """Demonstrate basic response synthesizers"""
        print("\n" + "="*80)
        print("BASIC RESPONSE SYNTHESIZERS DEMO")
        print("="*80)
        
        # Create sample nodes for demonstration
        sample_nodes = [
            NodeWithScore(
                node=TextNode(
                    text="Climate change is primarily caused by human activities, especially greenhouse gas emissions from burning fossil fuels. The main greenhouse gases include carbon dioxide, methane, and nitrous oxide.",
                    metadata={"source": "doc1"}
                ),
                score=0.95
            ),
            NodeWithScore(
                node=TextNode(
                    text="Carbon dioxide accounts for about 76% of total greenhouse gas emissions. The concentration of CO2 in the atmosphere has increased by over 40% since pre-industrial times.",
                    metadata={"source": "doc2"}
                ),
                score=0.88
            ),
            NodeWithScore(
                node=TextNode(
                    text="The global average temperature has risen by approximately 1.1°C since the late 1800s. Key indicators include rising sea levels, melting ice caps, and changing precipitation patterns.",
                    metadata={"source": "doc3"}
                ),
                score=0.82
            )
        ]
        
        query = "What are the main causes of climate change?"
        
        if not self.llm:
            print("LLM not available - showing synthesizer structure only")
            synthesizer_info = {
                "TreeSummarize": "Hierarchical summarization that builds responses bottom-up",
                "Accumulate": "Accumulates responses from individual sources", 
                "CompactAndRefine": "Compacts context and refines responses iteratively",
                "Refine": "Iteratively refines responses using retrieved context"
            }
            
            for name, description in synthesizer_info.items():
                print(f"\n{'-'*60}")
                print(f"SYNTHESIZER: {name}")
                print(f"Description: {description}")
                print(f"Would process query: {query}")
            return
        
        synthesizers = {
            "TreeSummarize": TreeSummarize(llm=self.llm),
            "Accumulate": Accumulate(llm=self.llm),
            "CompactAndRefine": CompactAndRefine(llm=self.llm),
            "Refine": get_response_synthesizer(response_mode=ResponseMode.REFINE, llm=self.llm)
        }
        
        for name, synthesizer in synthesizers.items():
            print(f"\n{'-'*60}")
            print(f"SYNTHESIZER: {name}")
            print(f"{'-'*60}")
            
            try:
                start_time = time.time()
                response = synthesizer.synthesize(query, sample_nodes)
                end_time = time.time()
                
                print(f"Query: {query}")
                print(f"Response: {response.response}")
                print(f"Processing time: {end_time - start_time:.2f}s")
                
                # Show source nodes if available
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    print(f"Source nodes used: {len(response.source_nodes)}")
                    for i, node in enumerate(response.source_nodes[:2]):  # Show first 2
                        print(f"  Node {i+1}: {node.node.text[:100]}...")
                
            except Exception as e:
                print(f"Error with {name}: {e}")
    
    def demo_response_modes(self):
        """Demonstrate different response modes"""
        print("\n" + "="*80)
        print("RESPONSE MODES DEMO")
        print("="*80)
        
        if not self.llm:
            print("LLM not available - showing response modes structure only")
            modes = [ResponseMode.REFINE, ResponseMode.COMPACT, ResponseMode.TREE_SUMMARIZE, 
                    ResponseMode.ACCUMULATE, ResponseMode.COMPACT_ACCUMULATE]
            for mode in modes:
                print(f"Response Mode: {mode.value} - Would process complex queries with different synthesis strategies")
            return
        
        if not self.index:
            print("Index not available, setting up...")
            if not self.setup_index():
                print("Cannot proceed without index")
                return
        
        query = "How does climate change affect ecosystems and what renewable energy solutions are available?"
        
        response_modes = [
            ResponseMode.REFINE,
            ResponseMode.COMPACT,
            ResponseMode.TREE_SUMMARIZE,
            ResponseMode.ACCUMULATE,
            ResponseMode.COMPACT_ACCUMULATE
        ]
        
        for mode in response_modes:
            print(f"\n{'-'*60}")
            print(f"RESPONSE MODE: {mode.value}")
            print(f"{'-'*60}")
            
            try:
                # Create query engine with specific response mode
                query_engine = self.index.as_query_engine(
                    response_mode=mode,
                    similarity_top_k=5,
                    llm=self.llm
                )
                
                start_time = time.time()
                response = query_engine.query(query)
                end_time = time.time()
                
                print(f"Query: {query}")
                print(f"Response: {response.response}")
                print(f"Processing time: {end_time - start_time:.2f}s")
                
                # Show source nodes info
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    print(f"Source nodes: {len(response.source_nodes)}")
                    for i, node in enumerate(response.source_nodes[:2]):
                        print(f"  Node {i+1} (Score: {node.score:.3f}): {node.node.text[:100]}...")
                
            except Exception as e:
                print(f"Error with {mode.value}: {e}")
    
    def demo_advanced_synthesizers(self):
        """Demonstrate advanced synthesizer configurations"""
        print("\n" + "="*80)
        print("ADVANCED SYNTHESIZER CONFIGURATIONS")
        print("="*80)
        
        if not self.index:
            print("Index not available, setting up...")
            if not self.setup_index():
                print("Cannot proceed without index")
                return
        
        query = "What are the comprehensive strategies for addressing climate change?"
        
        # Different synthesizer configurations
        configs = [
            {
                "name": "Tree Summarize (Detailed)",
                "synthesizer": get_response_synthesizer(
                    response_mode=ResponseMode.TREE_SUMMARIZE,
                    llm=self.llm,
                    use_async=False
                )
            },
            {
                "name": "Compact and Refine",
                "synthesizer": get_response_synthesizer(
                    response_mode=ResponseMode.COMPACT,
                    llm=self.llm,
                    use_async=False
                )
            },
            {
                "name": "Accumulate Responses",
                "synthesizer": get_response_synthesizer(
                    response_mode=ResponseMode.ACCUMULATE,
                    llm=self.llm,
                    use_async=False
                )
            }
        ]
        
        # Get relevant nodes for the query
        retriever = self.index.as_retriever(similarity_top_k=6)
        nodes = retriever.retrieve(query)
        
        print(f"Retrieved {len(nodes)} relevant nodes for synthesis")
        
        for config in configs:
            print(f"\n{'-'*60}")
            print(f"CONFIGURATION: {config['name']}")
            print(f"{'-'*60}")
            
            try:
                start_time = time.time()
                response = config["synthesizer"].synthesize(query, nodes)
                end_time = time.time()
                
                print(f"Query: {query}")
                print(f"Response: {response.response}")
                print(f"Processing time: {end_time - start_time:.2f}s")
                
                # Analyze response characteristics
                response_length = len(response.response.split())
                print(f"Response length: {response_length} words")
                
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    print(f"Source nodes utilized: {len(response.source_nodes)}")
                
            except Exception as e:
                print(f"Error with {config['name']}: {e}")
    
    def compare_synthesizer_performance(self):
        """Compare performance of different synthesizers"""
        print("\n" + "="*80)
        print("SYNTHESIZER PERFORMANCE COMPARISON")
        print("="*80)
        
        if not self.index:
            print("Index not available, setting up...")
            if not self.setup_index():
                print("Cannot proceed without index")
                return
        
        test_queries = [
            "What are the main causes of climate change?",
            "How do renewable energy technologies work?",
            "What adaptation strategies are available for climate change?"
        ]
        
        synthesizer_configs = [
            ("TreeSummarize", ResponseMode.TREE_SUMMARIZE),
            ("Accumulate", ResponseMode.ACCUMULATE),
            ("CompactAndRefine", ResponseMode.COMPACT),
            ("Refine", ResponseMode.REFINE)
        ]
        
        results = {}
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"QUERY: {query}")
            print(f"{'='*60}")
            
            results[query] = {}
            
            for name, mode in synthesizer_configs:
                try:
                    query_engine = self.index.as_query_engine(
                        response_mode=mode,
                        similarity_top_k=4,
                        llm=self.llm
                    )
                    
                    start_time = time.time()
                    response = query_engine.query(query)
                    end_time = time.time()
                    
                    processing_time = end_time - start_time
                    response_length = len(response.response.split())
                    
                    results[query][name] = {
                        "response": response.response,
                        "processing_time": processing_time,
                        "response_length": response_length,
                        "source_nodes": len(response.source_nodes) if hasattr(response, 'source_nodes') else 0
                    }
                    
                    print(f"\n{name}:")
                    print(f"  Time: {processing_time:.2f}s")
                    print(f"  Length: {response_length} words")
                    print(f"  Response: {response.response[:200]}...")
                    
                except Exception as e:
                    print(f"Error with {name}: {e}")
                    results[query][name] = {"error": str(e)}
        
        # Summary comparison
        print(f"\n{'='*60}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        
        for query, query_results in results.items():
            print(f"\nQuery: {query}")
            print("-" * 50)
            
            for synthesizer, result in query_results.items():
                if "error" not in result:
                    print(f"{synthesizer:15} | Time: {result['processing_time']:.2f}s | Words: {result['response_length']:3d} | Sources: {result['source_nodes']}")
                else:
                    print(f"{synthesizer:15} | Error: {result['error']}")
    
    def demo_custom_synthesis_strategies(self):
        """Demonstrate custom synthesis strategies"""
        print("\n" + "="*80)
        print("CUSTOM SYNTHESIS STRATEGIES")
        print("="*80)
        
        # Create custom prompt templates for different synthesis approaches
        custom_strategies = [
            {
                "name": "Analytical Synthesis",
                "system_prompt": "You are an analytical assistant. Synthesize the information by identifying key themes, analyzing relationships, and providing structured insights.",
                "response_mode": ResponseMode.TREE_SUMMARIZE
            },
            {
                "name": "Comparative Synthesis",
                "system_prompt": "You are a comparative analyst. Compare and contrast different viewpoints, highlight similarities and differences, and provide balanced perspectives.",
                "response_mode": ResponseMode.COMPACT
            },
            {
                "name": "Solution-Focused Synthesis",
                "system_prompt": "You are a solution-oriented advisor. Focus on actionable insights, practical recommendations, and implementable strategies.",
                "response_mode": ResponseMode.REFINE
            }
        ]
        
        if not self.index:
            print("Index not available, setting up...")
            if not self.setup_index():
                print("Cannot proceed without index")
                return
        
        query = "What are the most effective approaches to addressing climate change?"
        
        for strategy in custom_strategies:
            print(f"\n{'-'*60}")
            print(f"STRATEGY: {strategy['name']}")
            print(f"{'-'*60}")
            
            try:
                # Create LLM with custom system prompt
                # Note: Custom system prompts would need to be handled in the query engine
                custom_llm = AzureOpenAI(
                    model="gpt-4",
                    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
                    temperature=0.1,
                    max_tokens=1000,
                )
                
                query_engine = self.index.as_query_engine(
                    response_mode=strategy["response_mode"],
                    similarity_top_k=5,
                    llm=custom_llm
                )
                
                start_time = time.time()
                response = query_engine.query(query)
                end_time = time.time()
                
                print(f"System Prompt: {strategy['system_prompt']}")
                print(f"Response Mode: {strategy['response_mode'].value}")
                print(f"Query: {query}")
                print(f"Response: {response.response}")
                print(f"Processing time: {end_time - start_time:.2f}s")
                
            except Exception as e:
                print(f"Error with {strategy['name']}: {e}")
    
    def show_synthesis_best_practices(self):
        """Show best practices for response synthesis"""
        print("\n" + "="*80)
        print("RESPONSE SYNTHESIS BEST PRACTICES")
        print("="*80)
        
        practices = {
            "TreeSummarize": {
                "description": "Hierarchical summarization that builds responses bottom-up",
                "best_for": ["Complex queries requiring comprehensive analysis", "Multiple document synthesis", "Structured information organization"],
                "considerations": ["Higher computational cost", "Better for detailed responses", "Good for maintaining context across sources"]
            },
            "Accumulate": {
                "description": "Accumulates responses from individual sources",
                "best_for": ["Diverse perspectives on a topic", "Collecting multiple viewpoints", "Comprehensive coverage"],
                "considerations": ["May produce longer responses", "Good for getting different angles", "Less synthesis, more aggregation"]
            },
            "CompactAndRefine": {
                "description": "Compacts context and refines responses iteratively",
                "best_for": ["Balanced synthesis", "Moderate response length", "Efficient processing"],
                "considerations": ["Good balance of speed and quality", "Suitable for most use cases", "Moderate computational cost"]
            },
            "Refine": {
                "description": "Iteratively refines responses using retrieved context",
                "best_for": ["Detailed analysis", "Progressive refinement", "Context-aware responses"],
                "considerations": ["Sequential processing", "Good for deep analysis", "Higher processing time"]
            }
        }
        
        for name, info in practices.items():
            print(f"\n{name}:")
            print(f"  Description: {info['description']}")
            print(f"  Best for:")
            for use_case in info['best_for']:
                print(f"    - {use_case}")
            print(f"  Considerations:")
            for consideration in info['considerations']:
                print(f"    - {consideration}")
        
        print(f"\n{'='*60}")
        print("SELECTION GUIDELINES")
        print(f"{'='*60}")
        
        guidelines = [
            "Use TreeSummarize for comprehensive, well-structured responses",
            "Use Accumulate when you need multiple perspectives on a topic",
            "Use CompactAndRefine for balanced performance and quality",
            "Use Refine for detailed, context-aware analysis",
            "Consider response length requirements when choosing synthesizers",
            "Test different synthesizers with your specific use case",
            "Monitor processing time vs. quality trade-offs",
            "Adjust similarity_top_k based on your data and needs"
        ]
        
        for i, guideline in enumerate(guidelines, 1):
            print(f"{i:2d}. {guideline}")

def run_response_synthesizer_demo():
    """Run the complete response synthesizer demonstration"""
    print("\n" + "="*80)
    print("RESPONSE SYNTHESIZERS DEMO: LLAMAINDEX PROMPT SYNTHESIS")
    print("="*80)
    
    # Initialize demo
    demo = ResponseSynthesizerDemo()
    
    # Run basic synthesizers demo
    demo.demo_basic_synthesizers()
    
    # Run response modes demo
    demo.demo_response_modes()
    
    # Run advanced synthesizers demo
    demo.demo_advanced_synthesizers()
    
    # Compare synthesizer performance
    demo.compare_synthesizer_performance()
    
    # Show custom synthesis strategies
    demo.demo_custom_synthesis_strategies()
    
    # Show best practices
    demo.show_synthesis_best_practices()
    
    print("\n" + "="*80)
    print("RESPONSE SYNTHESIZERS DEMO COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    run_response_synthesizer_demo()