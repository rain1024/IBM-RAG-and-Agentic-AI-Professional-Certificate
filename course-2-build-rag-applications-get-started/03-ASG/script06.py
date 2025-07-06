"""
VectorStoreIndex Demo: LlamaIndex Vector Storage and Retrieval
================================================================================

Module: script06.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates various aspects of LlamaIndex VectorStoreIndex.
    It shows how to create, configure, and query vector indexes with different
    storage backends, retrieval strategies, and optimization techniques.
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import time
import numpy as np

# LlamaIndex imports
from llama_index.core import (
    Document, 
    VectorStoreIndex, 
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

# Load environment variables
load_dotenv()

class VectorStoreIndexDemo:
    """Demo class for exploring VectorStoreIndex features in LlamaIndex"""
    
    def __init__(self):
        self.sample_documents = self.create_sample_documents()
        self.sample_queries = self.create_sample_queries()
        self.llm = None
        self.embed_model = None
        self.indexes = {}
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
            
            # Set global settings
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            
            print("✓ Models initialized successfully")
            
        except Exception as e:
            print(f"⚠ Warning: Could not initialize models: {e}")
            print("Running in demo mode - will show structure without actual embedding/LLM calls")
            self.llm = None
            self.embed_model = None
    
    def create_sample_documents(self) -> List[Document]:
        """Create sample documents for demonstration"""
        documents = [
            Document(
                text="""
                Artificial Intelligence and Machine Learning Overview
                
                Artificial Intelligence (AI) refers to the simulation of human intelligence in machines 
                that are programmed to think and learn like humans. Machine Learning (ML) is a subset 
                of AI that enables computers to learn and improve from experience without being explicitly 
                programmed for every task.
                
                Key areas of AI include natural language processing, computer vision, robotics, and 
                expert systems. Machine learning techniques include supervised learning, unsupervised 
                learning, and reinforcement learning. Deep learning, a subset of machine learning, 
                uses neural networks with multiple layers to model complex patterns in data.
                
                AI applications span across industries including healthcare, finance, transportation, 
                and entertainment. Common examples include recommendation systems, image recognition, 
                voice assistants, and autonomous vehicles.
                """,
                metadata={"source": "ai_overview", "category": "technology", "domain": "artificial_intelligence"}
            ),
            Document(
                text="""
                Natural Language Processing Applications
                
                Natural Language Processing (NLP) is a branch of AI that focuses on the interaction 
                between computers and human language. NLP combines computational linguistics with 
                statistical and machine learning models to enable computers to understand, interpret, 
                and generate human language.
                
                Key NLP tasks include text classification, sentiment analysis, named entity recognition, 
                part-of-speech tagging, and machine translation. Recent advances in transformer 
                architectures, such as BERT and GPT models, have significantly improved NLP performance 
                across various tasks.
                
                NLP applications include chatbots, language translation services, text summarization, 
                spam detection, and content analysis. The field continues to evolve with large 
                language models showing remarkable capabilities in understanding context and 
                generating coherent text.
                """,
                metadata={"source": "nlp_applications", "category": "technology", "domain": "natural_language_processing"}
            ),
            Document(
                text="""
                Computer Vision and Image Processing
                
                Computer Vision is a field of AI that enables computers to interpret and understand 
                visual information from the world. It involves acquiring, processing, analyzing, 
                and understanding digital images and videos to extract meaningful information.
                
                Core computer vision tasks include image classification, object detection, semantic 
                segmentation, and image generation. Convolutional Neural Networks (CNNs) have been 
                particularly successful in computer vision applications, with architectures like 
                ResNet, VGG, and more recently, Vision Transformers.
                
                Applications of computer vision include medical imaging, autonomous vehicles, 
                facial recognition, quality control in manufacturing, and augmented reality. 
                The field has seen rapid advancement with the availability of large datasets 
                and improved computational resources.
                """,
                metadata={"source": "computer_vision", "category": "technology", "domain": "computer_vision"}
            ),
            Document(
                text="""
                Robotics and Automation
                
                Robotics combines AI, mechanical engineering, and computer science to create 
                intelligent machines capable of performing tasks typically requiring human 
                intelligence and dexterity. Modern robotics heavily relies on AI for perception, 
                decision-making, and adaptive behavior.
                
                Key areas in robotics include manipulation, navigation, human-robot interaction, 
                and swarm robotics. Sensors, actuators, and control systems work together to 
                enable robots to perceive their environment and execute complex tasks. Machine 
                learning algorithms help robots adapt to new situations and improve performance.
                
                Robotics applications span manufacturing, healthcare, exploration, service industries, 
                and domestic assistance. Industrial robots have revolutionized manufacturing 
                processes, while service robots are increasingly used in healthcare, cleaning, 
                and customer service applications.
                """,
                metadata={"source": "robotics", "category": "technology", "domain": "robotics"}
            ),
            Document(
                text="""
                Data Science and Analytics
                
                Data Science is an interdisciplinary field that uses scientific methods, algorithms, 
                and systems to extract knowledge and insights from structured and unstructured data. 
                It combines statistics, computer science, and domain expertise to analyze complex 
                datasets and make data-driven decisions.
                
                The data science process typically involves data collection, cleaning, exploration, 
                modeling, and interpretation. Common techniques include statistical analysis, 
                machine learning, data visualization, and predictive modeling. Tools and languages 
                commonly used include Python, R, SQL, and various specialized libraries.
                
                Data science applications include business intelligence, predictive analytics, 
                customer segmentation, fraud detection, and recommendation systems. The field 
                continues to grow with the increasing availability of data and computational 
                resources across industries.
                """,
                metadata={"source": "data_science", "category": "technology", "domain": "data_science"}
            ),
            Document(
                text="""
                Cloud Computing and Distributed Systems
                
                Cloud Computing provides on-demand access to computing resources including servers, 
                storage, databases, networking, and software over the internet. It enables scalable, 
                flexible, and cost-effective IT infrastructure without requiring physical hardware 
                maintenance.
                
                Key cloud service models include Infrastructure as a Service (IaaS), Platform as 
                a Service (PaaS), and Software as a Service (SaaS). Major cloud providers include 
                Amazon Web Services, Microsoft Azure, and Google Cloud Platform, each offering 
                comprehensive suites of services.
                
                Cloud computing benefits include cost reduction, scalability, accessibility, 
                and disaster recovery. Applications range from simple web hosting to complex 
                enterprise solutions, big data processing, and machine learning model deployment. 
                The field continues to evolve with serverless computing and edge computing trends.
                """,
                metadata={"source": "cloud_computing", "category": "technology", "domain": "cloud_computing"}
            )
        ]
        
        return documents
    
    def create_sample_queries(self) -> List[str]:
        """Create sample queries for testing vector search"""
        return [
            "What is machine learning and how does it work?",
            "How are neural networks used in computer vision?",
            "What are the applications of natural language processing?",
            "How do robots use AI for navigation and manipulation?",
            "What tools are commonly used in data science?",
            "What are the benefits of cloud computing?",
            "How does deep learning differ from traditional machine learning?",
            "What are the main challenges in computer vision?",
            "How is AI applied in healthcare and medicine?",
            "What are the key components of a data science pipeline?"
        ]
    
    def demo_basic_vector_index_creation(self):
        """Demonstrate basic vector index creation"""
        print("\n" + "="*80)
        print("BASIC VECTOR INDEX CREATION DEMO")
        print("="*80)
        
        if not self.embed_model:
            print("Embedding model not available - showing structure only")
            print("Would create VectorStoreIndex from documents with embeddings")
            return
        
        try:
            # Create basic vector index
            start_time = time.time()
            basic_index = VectorStoreIndex.from_documents(
                self.sample_documents,
                embed_model=self.embed_model
            )
            end_time = time.time()
            
            self.indexes["basic"] = basic_index
            
            print(f"✓ Created basic vector index in {end_time - start_time:.2f}s")
            print(f"Documents indexed: {len(self.sample_documents)}")
            
            # Show index structure
            print("\nIndex Information:")
            print(f"  - Index Type: {type(basic_index).__name__}")
            print(f"  - Vector Store: {type(basic_index.vector_store).__name__}")
            print(f"  - Embedding Model: {type(self.embed_model).__name__}")
            
            # Test basic query
            query = "What is artificial intelligence?"
            query_engine = basic_index.as_query_engine()
            
            print(f"\nTesting basic query: {query}")
            start_time = time.time()
            response = query_engine.query(query)
            end_time = time.time()
            
            print(f"Response: {response.response[:200]}...")
            print(f"Query time: {end_time - start_time:.2f}s")
            
        except Exception as e:
            print(f"Error creating basic index: {e}")
    
    def demo_vector_index_with_custom_nodes(self):
        """Demonstrate vector index creation with custom nodes"""
        print("\n" + "="*80)
        print("VECTOR INDEX WITH CUSTOM NODES DEMO")
        print("="*80)
        
        if not self.embed_model:
            print("Embedding model not available - showing structure only")
            return
        
        try:
            # Create custom nodes with different chunk sizes
            node_parser = SentenceSplitter(
                chunk_size=300, 
                chunk_overlap=50,
                separator=" "
            )
            
            nodes = node_parser.get_nodes_from_documents(self.sample_documents)
            
            # Add custom metadata to nodes
            for i, node in enumerate(nodes):
                node.metadata.update({
                    "chunk_id": f"chunk_{i}",
                    "chunk_size": len(node.text),
                    "word_count": len(node.text.split())
                })
            
            print(f"Created {len(nodes)} custom nodes")
            print(f"Average chunk size: {np.mean([len(node.text) for node in nodes]):.1f} characters")
            
            # Create index from custom nodes
            start_time = time.time()
            custom_index = VectorStoreIndex(nodes, embed_model=self.embed_model)
            end_time = time.time()
            
            self.indexes["custom_nodes"] = custom_index
            
            print(f"✓ Created custom node index in {end_time - start_time:.2f}s")
            
            # Test retrieval with custom nodes
            query = "How does machine learning work?"
            retriever = custom_index.as_retriever(similarity_top_k=3)
            
            print(f"\nTesting retrieval: {query}")
            retrieved_nodes = retriever.retrieve(query)
            
            print(f"Retrieved {len(retrieved_nodes)} nodes:")
            for i, node in enumerate(retrieved_nodes):
                print(f"  Node {i+1} (Score: {node.score:.3f}):")
                print(f"    Chunk ID: {node.metadata.get('chunk_id', 'N/A')}")
                print(f"    Word Count: {node.metadata.get('word_count', 'N/A')}")
                print(f"    Text: {node.text[:100]}...")
                print()
            
        except Exception as e:
            print(f"Error creating custom node index: {e}")
    
    def demo_vector_index_with_storage_context(self):
        """Demonstrate vector index with custom storage context"""
        print("\n" + "="*80)
        print("VECTOR INDEX WITH STORAGE CONTEXT DEMO")
        print("="*80)
        
        if not self.embed_model:
            print("Embedding model not available - showing structure only")
            return
        
        try:
            # Create storage context with custom vector store
            vector_store = SimpleVectorStore()
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                docstore=SimpleDocumentStore(),
                index_store=SimpleIndexStore()
            )
            
            print("Created custom storage context:")
            print(f"  - Vector Store: {type(vector_store).__name__}")
            print(f"  - Doc Store: {type(storage_context.docstore).__name__}")
            print(f"  - Index Store: {type(storage_context.index_store).__name__}")
            
            # Create index with storage context
            start_time = time.time()
            storage_index = VectorStoreIndex.from_documents(
                self.sample_documents,
                storage_context=storage_context,
                embed_model=self.embed_model
            )
            end_time = time.time()
            
            self.indexes["storage_context"] = storage_index
            
            print(f"✓ Created storage context index in {end_time - start_time:.2f}s")
            
            # Show storage information
            print("\nStorage Information:")
            print(f"  - Documents in docstore: {len(storage_context.docstore.docs)}")
            print(f"  - Vector store data: {len(vector_store.data) if hasattr(vector_store, 'data') else 'N/A'}")
            
            # Test different retrieval configurations
            retrieval_configs = [
                {"similarity_top_k": 2, "name": "Top 2"},
                {"similarity_top_k": 5, "name": "Top 5"},
                {"similarity_top_k": 3, "name": "Top 3"}
            ]
            
            query = "What are the applications of computer vision?"
            
            for config in retrieval_configs:
                print(f"\n{config['name']} Retrieval:")
                retriever = storage_index.as_retriever(**{k: v for k, v in config.items() if k != 'name'})
                nodes = retriever.retrieve(query)
                
                print(f"  Retrieved {len(nodes)} nodes")
                for i, node in enumerate(nodes):
                    print(f"    Node {i+1}: Score {node.score:.3f}, Domain: {node.metadata.get('domain', 'N/A')}")
            
        except Exception as e:
            print(f"Error creating storage context index: {e}")
    
    def demo_vector_index_querying_strategies(self):
        """Demonstrate different querying strategies"""
        print("\n" + "="*80)
        print("VECTOR INDEX QUERYING STRATEGIES DEMO")
        print("="*80)
        
        if "basic" not in self.indexes:
            print("Basic index not available, creating...")
            self.demo_basic_vector_index_creation()
        
        if "basic" not in self.indexes:
            print("Cannot proceed without basic index")
            return
        
        index = self.indexes["basic"]
        query = "How is AI used in healthcare applications?"
        
        # Different query engine configurations
        query_configs = [
            {
                "name": "Default Query Engine",
                "config": {}
            },
            {
                "name": "High Similarity Threshold",
                "config": {
                    "similarity_top_k": 3,
                    "node_postprocessors": [SimilarityPostprocessor(similarity_cutoff=0.8)]
                }
            },
            {
                "name": "Low Similarity Threshold",
                "config": {
                    "similarity_top_k": 5,
                    "node_postprocessors": [SimilarityPostprocessor(similarity_cutoff=0.3)]
                }
            },
            {
                "name": "Large Context Window",
                "config": {
                    "similarity_top_k": 8
                }
            }
        ]
        
        for config in query_configs:
            print(f"\n{'-'*60}")
            print(f"STRATEGY: {config['name']}")
            print(f"{'-'*60}")
            
            try:
                query_engine = index.as_query_engine(**config["config"])
                
                start_time = time.time()
                response = query_engine.query(query)
                end_time = time.time()
                
                print(f"Query: {query}")
                print(f"Response: {response.response[:300]}...")
                print(f"Query time: {end_time - start_time:.2f}s")
                
                # Show source information
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    print(f"Source nodes: {len(response.source_nodes)}")
                    for i, node in enumerate(response.source_nodes[:3]):
                        print(f"  Source {i+1}: Score {node.score:.3f}, Domain: {node.metadata.get('domain', 'N/A')}")
                
            except Exception as e:
                print(f"Error with {config['name']}: {e}")
    
    def demo_vector_index_retrieval_modes(self):
        """Demonstrate different retrieval modes"""
        print("\n" + "="*80)
        print("VECTOR INDEX RETRIEVAL MODES DEMO")
        print("="*80)
        
        if "basic" not in self.indexes:
            print("Basic index not available, creating...")
            self.demo_basic_vector_index_creation()
        
        if "basic" not in self.indexes:
            print("Cannot proceed without basic index")
            return
        
        index = self.indexes["basic"]
        query = "What are the key components of data science?"
        
        # Different retrieval configurations
        retrieval_configs = [
            {
                "name": "Default Retrieval",
                "retriever": index.as_retriever(similarity_top_k=4)
            },
            {
                "name": "High Precision Retrieval",
                "retriever": index.as_retriever(
                    similarity_top_k=2,
                    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.85)]
                )
            },
            {
                "name": "High Recall Retrieval",
                "retriever": index.as_retriever(
                    similarity_top_k=8,
                    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.1)]
                )
            }
        ]
        
        for config in retrieval_configs:
            print(f"\n{'-'*60}")
            print(f"MODE: {config['name']}")
            print(f"{'-'*60}")
            
            try:
                start_time = time.time()
                nodes = config["retriever"].retrieve(query)
                end_time = time.time()
                
                print(f"Query: {query}")
                print(f"Retrieved {len(nodes)} nodes in {end_time - start_time:.2f}s")
                
                # Analyze retrieved nodes
                if nodes:
                    scores = [node.score for node in nodes]
                    print(f"Score statistics:")
                    print(f"  - Max score: {max(scores):.3f}")
                    print(f"  - Min score: {min(scores):.3f}")
                    print(f"  - Avg score: {np.mean(scores):.3f}")
                    
                    # Show top nodes
                    print(f"Top retrieved nodes:")
                    for i, node in enumerate(nodes[:3]):
                        print(f"  Node {i+1}:")
                        print(f"    Score: {node.score:.3f}")
                        print(f"    Domain: {node.metadata.get('domain', 'N/A')}")
                        print(f"    Text: {node.text[:100]}...")
                        print()
                
            except Exception as e:
                print(f"Error with {config['name']}: {e}")
    
    def demo_vector_index_performance_analysis(self):
        """Demonstrate performance analysis of vector indexes"""
        print("\n" + "="*80)
        print("VECTOR INDEX PERFORMANCE ANALYSIS")
        print("="*80)
        
        if "basic" not in self.indexes:
            print("Basic index not available, creating...")
            self.demo_basic_vector_index_creation()
        
        if "basic" not in self.indexes:
            print("Cannot proceed without basic index")
            return
        
        index = self.indexes["basic"]
        test_queries = [
            "What is machine learning?",
            "How does computer vision work?",
            "What are NLP applications?",
            "How is AI used in robotics?",
            "What is cloud computing?"
        ]
        
        # Test different similarity_top_k values
        top_k_values = [1, 3, 5, 10]
        
        print("Performance analysis across different top_k values:")
        print(f"{'top_k':<6} {'avg_time':<10} {'avg_nodes':<12} {'avg_score':<12}")
        print("-" * 50)
        
        for top_k in top_k_values:
            times = []
            node_counts = []
            scores = []
            
            retriever = index.as_retriever(similarity_top_k=top_k)
            
            for query in test_queries:
                start_time = time.time()
                nodes = retriever.retrieve(query)
                end_time = time.time()
                
                times.append(end_time - start_time)
                node_counts.append(len(nodes))
                if nodes:
                    scores.extend([node.score for node in nodes])
            
            avg_time = np.mean(times)
            avg_nodes = np.mean(node_counts)
            avg_score = np.mean(scores) if scores else 0
            
            print(f"{top_k:<6} {avg_time:<10.3f} {avg_nodes:<12.1f} {avg_score:<12.3f}")
        
        # Test query complexity impact
        print(f"\nQuery complexity analysis:")
        query_types = [
            ("Simple", "AI"),
            ("Medium", "What is machine learning?"),
            ("Complex", "How do neural networks work in computer vision applications?"),
            ("Very Complex", "What are the differences between supervised and unsupervised learning in the context of natural language processing applications?")
        ]
        
        print(f"{'complexity':<12} {'query_time':<12} {'nodes':<8} {'response_length':<15}")
        print("-" * 50)
        
        for complexity, query in query_types:
            try:
                query_engine = index.as_query_engine(similarity_top_k=5)
                
                start_time = time.time()
                response = query_engine.query(query)
                end_time = time.time()
                
                query_time = end_time - start_time
                nodes = len(response.source_nodes) if hasattr(response, 'source_nodes') else 0
                response_length = len(response.response.split())
                
                print(f"{complexity:<12} {query_time:<12.3f} {nodes:<8} {response_length:<15}")
                
            except Exception as e:
                print(f"{complexity:<12} ERROR: {str(e)[:30]}...")
    
    def demo_vector_index_metadata_filtering(self):
        """Demonstrate metadata filtering in vector indexes"""
        print("\n" + "="*80)
        print("VECTOR INDEX METADATA FILTERING DEMO")
        print("="*80)
        
        if not self.embed_model:
            print("Embedding model not available - showing structure only")
            return
        
        try:
            # Create index with rich metadata
            enriched_docs = []
            for doc in self.sample_documents:
                doc.metadata.update({
                    "text_length": len(doc.text),
                    "word_count": len(doc.text.split()),
                    "has_code": "import" in doc.text or "function" in doc.text,
                    "complexity": "high" if len(doc.text.split()) > 200 else "medium" if len(doc.text.split()) > 100 else "low"
                })
                enriched_docs.append(doc)
            
            metadata_index = VectorStoreIndex.from_documents(
                enriched_docs,
                embed_model=self.embed_model
            )
            
            self.indexes["metadata"] = metadata_index
            
            # Test queries with different metadata considerations
            query = "What are the applications of AI?"
            
            # Show all available metadata
            print("Available metadata fields:")
            all_metadata = {}
            for doc in enriched_docs:
                for key, value in doc.metadata.items():
                    if key not in all_metadata:
                        all_metadata[key] = set()
                    all_metadata[key].add(str(value))
            
            for key, values in all_metadata.items():
                print(f"  {key}: {', '.join(sorted(values))}")
            
            # Test retrieval with different approaches
            retrieval_approaches = [
                {"name": "All Documents", "top_k": 5},
                {"name": "High Complexity Only", "top_k": 3},
                {"name": "Technology Domain", "top_k": 4}
            ]
            
            for approach in retrieval_approaches:
                print(f"\n{'-'*50}")
                print(f"APPROACH: {approach['name']}")
                print(f"{'-'*50}")
                
                retriever = metadata_index.as_retriever(similarity_top_k=approach["top_k"])
                nodes = retriever.retrieve(query)
                
                print(f"Retrieved {len(nodes)} nodes:")
                for i, node in enumerate(nodes):
                    print(f"  Node {i+1}:")
                    print(f"    Score: {node.score:.3f}")
                    print(f"    Domain: {node.metadata.get('domain', 'N/A')}")
                    print(f"    Complexity: {node.metadata.get('complexity', 'N/A')}")
                    print(f"    Word Count: {node.metadata.get('word_count', 'N/A')}")
                    print(f"    Text: {node.text[:80]}...")
                    print()
            
        except Exception as e:
            print(f"Error with metadata filtering: {e}")
    
    def demo_vector_index_best_practices(self):
        """Demonstrate best practices for vector indexes"""
        print("\n" + "="*80)
        print("VECTOR INDEX BEST PRACTICES")
        print("="*80)
        
        practices = {
            "Document Preparation": [
                "Clean and preprocess text before indexing",
                "Add meaningful metadata for better filtering",
                "Consider document chunking strategies",
                "Remove or handle special characters appropriately"
            ],
            "Index Configuration": [
                "Choose appropriate chunk size (200-500 tokens typically)",
                "Set reasonable chunk overlap (10-20%)",
                "Use consistent embedding models",
                "Configure storage context based on scale"
            ],
            "Retrieval Optimization": [
                "Tune similarity_top_k based on use case",
                "Use similarity cutoff to filter low-quality results",
                "Consider post-processing for result refinement",
                "Monitor and adjust based on performance metrics"
            ],
            "Performance Considerations": [
                "Index incrementally for large datasets",
                "Use appropriate vector store backend for scale",
                "Monitor embedding computation costs",
                "Consider caching for frequently accessed data"
            ],
            "Query Optimization": [
                "Craft clear and specific queries",
                "Use metadata filters when appropriate",
                "Consider query expansion techniques",
                "Test with representative query patterns"
            ]
        }
        
        for category, items in practices.items():
            print(f"\n{category}:")
            for item in items:
                print(f"  • {item}")
        
        # Performance comparison example
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON EXAMPLE")
        print(f"{'='*60}")
        
        if "basic" in self.indexes:
            index = self.indexes["basic"]
            
            # Test different configurations
            configs = [
                {"name": "Conservative", "top_k": 3, "cutoff": 0.8},
                {"name": "Balanced", "top_k": 5, "cutoff": 0.6},
                {"name": "Comprehensive", "top_k": 8, "cutoff": 0.3}
            ]
            
            query = "How does machine learning work?"
            
            print(f"{'Config':<15} {'Nodes':<8} {'Time':<8} {'Avg Score':<12}")
            print("-" * 50)
            
            for config in configs:
                try:
                    retriever = index.as_retriever(
                        similarity_top_k=config["top_k"],
                        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=config["cutoff"])]
                    )
                    
                    start_time = time.time()
                    nodes = retriever.retrieve(query)
                    end_time = time.time()
                    
                    avg_score = np.mean([node.score for node in nodes]) if nodes else 0
                    
                    print(f"{config['name']:<15} {len(nodes):<8} {end_time-start_time:<8.3f} {avg_score:<12.3f}")
                    
                except Exception as e:
                    print(f"{config['name']:<15} ERROR: {str(e)[:20]}...")
    
    def show_vector_index_summary(self):
        """Show summary of vector index capabilities"""
        print("\n" + "="*80)
        print("VECTOR INDEX SUMMARY")
        print("="*80)
        
        summary = {
            "Key Features": [
                "Semantic search using vector embeddings",
                "Flexible document and node handling",
                "Configurable retrieval strategies",
                "Metadata filtering capabilities",
                "Multiple storage backend support"
            ],
            "Use Cases": [
                "Document search and retrieval",
                "Question answering systems",
                "Recommendation systems",
                "Content similarity analysis",
                "Knowledge base applications"
            ],
            "Configuration Options": [
                "similarity_top_k: Number of results to retrieve",
                "chunk_size: Size of text chunks for processing",
                "chunk_overlap: Overlap between chunks",
                "embedding_model: Model for generating embeddings",
                "storage_context: Custom storage configuration"
            ],
            "Performance Tips": [
                "Use appropriate chunk sizes for your content",
                "Implement metadata filtering for better precision",
                "Monitor embedding costs and query performance",
                "Consider incremental indexing for large datasets",
                "Use similarity cutoffs to filter low-quality results"
            ]
        }
        
        for section, items in summary.items():
            print(f"\n{section}:")
            for item in items:
                print(f"  • {item}")
        
        # Show created indexes
        print(f"\n{'='*60}")
        print("CREATED INDEXES IN THIS DEMO")
        print(f"{'='*60}")
        
        if self.indexes:
            for name, index in self.indexes.items():
                print(f"\n{name.upper()} INDEX:")
                print(f"  Type: {type(index).__name__}")
                print(f"  Vector Store: {type(index.vector_store).__name__}")
                # Additional index information could be shown here
        else:
            print("No indexes were created (likely due to missing embedding model)")

def run_vector_store_index_demo():
    """Run the complete vector store index demonstration"""
    print("\n" + "="*80)
    print("VECTOR STORE INDEX DEMO: LLAMAINDEX VECTOR STORAGE AND RETRIEVAL")
    print("="*80)
    
    # Initialize demo
    demo = VectorStoreIndexDemo()
    
    # Run basic vector index creation
    demo.demo_basic_vector_index_creation()
    
    # Run custom nodes demo
    demo.demo_vector_index_with_custom_nodes()
    
    # Run storage context demo
    demo.demo_vector_index_with_storage_context()
    
    # Run querying strategies demo
    demo.demo_vector_index_querying_strategies()
    
    # Run retrieval modes demo
    demo.demo_vector_index_retrieval_modes()
    
    # Run performance analysis
    demo.demo_vector_index_performance_analysis()
    
    # Run metadata filtering demo
    demo.demo_vector_index_metadata_filtering()
    
    # Show best practices
    demo.demo_vector_index_best_practices()
    
    # Show summary
    demo.show_vector_index_summary()
    
    print("\n" + "="*80)
    print("VECTOR STORE INDEX DEMO COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    run_vector_store_index_demo()