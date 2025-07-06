"""
Text Splitters Demo: LangChain vs LlamaIndex Comparison
================================================================================

Module: script04.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates various text splitters from LangChain and LlamaIndex.
    It compares how different splitters chunk text including SentenceSplitter,
    RecursiveCharacterTextSplitter, CharacterTextSplitter, SemanticChunker,
    and shows their differences in chunking strategies.
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import textwrap

# LangChain imports
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter
)

# LlamaIndex imports
from llama_index.core.node_parser import (
    SentenceSplitter,
    TokenTextSplitter as LlamaTokenTextSplitter,
    CodeSplitter,
    MarkdownNodeParser,
    SimpleNodeParser
)
from llama_index.core import Document
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser

# Load environment variables
load_dotenv()

class TextSplitterDemo:
    """Demo class for comparing different text splitters"""
    
    def __init__(self):
        self.sample_texts = self.create_sample_texts()
        self.embedding_model = None
        self.initialize_embeddings()
    
    def initialize_embeddings(self):
        """Initialize embeddings for semantic chunker"""
        try:
            self.embedding_model = AzureOpenAIEmbedding(
                model="text-embedding-ada-002",
                deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
            )
            print("✓ Embeddings initialized successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize embeddings: {e}")
            print("Semantic chunker will be skipped in demo")
    
    def create_sample_texts(self) -> Dict[str, str]:
        """Create sample texts for different types of content"""
        return {
            "business_document": """
            Company Policy Document: Remote Work Guidelines
            
            Section 1: Introduction
            This document outlines the company's remote work policy effective January 2024. 
            All employees must familiarize themselves with these guidelines to ensure 
            compliance and maintain productivity standards.
            
            Section 2: Eligibility Criteria
            Remote work eligibility depends on several factors including job role, 
            performance history, and manager approval. Full-time employees with at least 
            six months of tenure are eligible to apply for remote work arrangements.
            
            Section 3: Equipment and Technology
            The company provides necessary equipment including laptops, monitors, and 
            communication tools. Employees are responsible for maintaining a secure 
            and professional workspace. Internet connectivity requirements include 
            minimum 25 Mbps download speed and reliable connection.
            
            Section 4: Working Hours and Availability
            Remote workers must maintain core business hours from 9 AM to 3 PM local time. 
            Flexibility is allowed for start and end times, but total daily hours must 
            meet minimum requirements. Regular check-ins with managers are mandatory.
            
            Section 5: Performance and Accountability
            Performance metrics remain unchanged for remote workers. Regular evaluations 
            will assess productivity, communication effectiveness, and goal achievement. 
            Failure to meet standards may result in return to office requirements.
            """,
            
            "technical_content": """
            # Python Data Processing Tutorial
            
            ## Introduction to Data Processing
            Data processing is a fundamental skill in modern programming. This tutorial 
            covers essential techniques for cleaning, transforming, and analyzing data.
            
            ```python
            import pandas as pd
            import numpy as np
            
            # Load data from CSV file
            df = pd.read_csv('data.csv')
            
            # Basic data exploration
            print(df.head())
            print(df.info())
            print(df.describe())
            ```
            
            ## Data Cleaning Techniques
            Data cleaning involves identifying and correcting errors, handling missing 
            values, and standardizing formats. Common issues include duplicate records, 
            inconsistent naming conventions, and outliers.
            
            ```python
            # Remove duplicates
            df_clean = df.drop_duplicates()
            
            # Handle missing values
            df_clean = df_clean.fillna(method='forward')
            
            # Standardize column names
            df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
            ```
            
            ## Data Transformation
            Transformation involves converting data into suitable formats for analysis. 
            This includes type conversions, scaling, and feature engineering.
            """,
            
            "conversational_text": """
            Hey Sarah! I hope you're doing well. I wanted to follow up on our conversation 
            yesterday about the project timeline. As we discussed, there are several 
            important deadlines we need to meet.
            
            First, the initial research phase needs to be completed by next Friday. 
            This includes market analysis, competitor research, and user interviews. 
            I know it's a tight deadline, but I believe we can make it work.
            
            Second, we need to finalize the design mockups by the end of the month. 
            The design team has been working hard on this, and I think we're on track. 
            However, we might need to adjust some specifications based on the research findings.
            
            Third, development should start in early February. We'll need to coordinate 
            with the engineering team to ensure smooth handoffs. I suggest we schedule 
            a kickoff meeting once the designs are approved.
            
            Let me know if you have any questions or concerns. I'm happy to discuss 
            this further in our next team meeting.
            
            Best regards,
            Alex
            """
        }
    
    def demo_langchain_splitters(self):
        """Demonstrate LangChain text splitters"""
        print("\n" + "="*60)
        print("LANGCHAIN TEXT SPLITTERS DEMO")
        print("="*60)
        
        # 1. RecursiveCharacterTextSplitter
        print("\n1. RecursiveCharacterTextSplitter")
        print("-" * 40)
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        recursive_chunks = recursive_splitter.split_text(self.sample_texts["business_document"])
        print(f"Number of chunks: {len(recursive_chunks)}")
        for i, chunk in enumerate(recursive_chunks[:3]):  # Show first 3 chunks
            print(f"\nChunk {i+1} (length: {len(chunk)}):")
            print(textwrap.fill(chunk.strip(), width=80))
        
        # 2. CharacterTextSplitter
        print("\n\n2. CharacterTextSplitter")
        print("-" * 40)
        char_splitter = CharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separator="\n\n"
        )
        
        char_chunks = char_splitter.split_text(self.sample_texts["business_document"])
        print(f"Number of chunks: {len(char_chunks)}")
        for i, chunk in enumerate(char_chunks[:3]):
            print(f"\nChunk {i+1} (length: {len(chunk)}):")
            print(textwrap.fill(chunk.strip(), width=80))
        
        # 3. TokenTextSplitter
        print("\n\n3. TokenTextSplitter")
        print("-" * 40)
        token_splitter = TokenTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )
        
        token_chunks = token_splitter.split_text(self.sample_texts["conversational_text"])
        print(f"Number of chunks: {len(token_chunks)}")
        for i, chunk in enumerate(token_chunks[:3]):
            print(f"\nChunk {i+1} (length: {len(chunk)}):")
            print(textwrap.fill(chunk.strip(), width=80))
        
        # 4. MarkdownTextSplitter
        print("\n\n4. MarkdownTextSplitter")
        print("-" * 40)
        markdown_splitter = MarkdownTextSplitter(
            chunk_size=200,
            chunk_overlap=30
        )
        
        markdown_chunks = markdown_splitter.split_text(self.sample_texts["technical_content"])
        print(f"Number of chunks: {len(markdown_chunks)}")
        for i, chunk in enumerate(markdown_chunks[:3]):
            print(f"\nChunk {i+1} (length: {len(chunk)}):")
            print(textwrap.fill(chunk.strip(), width=80))
        
        # 5. PythonCodeTextSplitter
        print("\n\n5. PythonCodeTextSplitter")
        print("-" * 40)
        python_splitter = PythonCodeTextSplitter(
            chunk_size=200,
            chunk_overlap=30
        )
        
        python_chunks = python_splitter.split_text(self.sample_texts["technical_content"])
        print(f"Number of chunks: {len(python_chunks)}")
        for i, chunk in enumerate(python_chunks[:3]):
            print(f"\nChunk {i+1} (length: {len(chunk)}):")
            print(textwrap.fill(chunk.strip(), width=80))
    
    def demo_llamaindex_splitters(self):
        """Demonstrate LlamaIndex text splitters"""
        print("\n" + "="*60)
        print("LLAMAINDEX TEXT SPLITTERS DEMO")
        print("="*60)
        
        # 1. SentenceSplitter
        print("\n1. SentenceSplitter")
        print("-" * 40)
        sentence_splitter = SentenceSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        
        doc = Document(text=self.sample_texts["business_document"])
        sentence_nodes = sentence_splitter.get_nodes_from_documents([doc])
        print(f"Number of nodes: {len(sentence_nodes)}")
        for i, node in enumerate(sentence_nodes[:3]):
            print(f"\nNode {i+1} (length: {len(node.text)}):")
            print(textwrap.fill(node.text.strip(), width=80))
        
        # 2. SimpleNodeParser
        print("\n\n2. SimpleNodeParser")
        print("-" * 40)
        simple_parser = SimpleNodeParser.from_defaults(
            chunk_size=300,
            chunk_overlap=50
        )
        
        simple_nodes = simple_parser.get_nodes_from_documents([doc])
        print(f"Number of nodes: {len(simple_nodes)}")
        for i, node in enumerate(simple_nodes[:3]):
            print(f"\nNode {i+1} (length: {len(node.text)}):")
            print(textwrap.fill(node.text.strip(), width=80))
        
        # 3. TokenTextSplitter (LlamaIndex)
        print("\n\n3. TokenTextSplitter (LlamaIndex)")
        print("-" * 40)
        llama_token_splitter = LlamaTokenTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )
        
        token_nodes = llama_token_splitter.get_nodes_from_documents([doc])
        print(f"Number of nodes: {len(token_nodes)}")
        for i, node in enumerate(token_nodes[:3]):
            print(f"\nNode {i+1} (length: {len(node.text)}):")
            print(textwrap.fill(node.text.strip(), width=80))
        
        # 4. CodeSplitter
        print("\n\n4. CodeSplitter")
        print("-" * 40)
        try:
            code_splitter = CodeSplitter(
                language="python",
                chunk_lines=10,
                chunk_lines_overlap=2
            )
            
            code_doc = Document(text=self.sample_texts["technical_content"])
            code_nodes = code_splitter.get_nodes_from_documents([code_doc])
            print(f"Number of nodes: {len(code_nodes)}")
            for i, node in enumerate(code_nodes[:3]):
                print(f"\nNode {i+1} (length: {len(node.text)}):")
                print(textwrap.fill(node.text.strip(), width=80))
        except Exception as e:
            print(f"CodeSplitter not available: {e}")
            print("Install tree-sitter and tree-sitter-python to use CodeSplitter")
        
        # 5. MarkdownNodeParser
        print("\n\n5. MarkdownNodeParser")
        print("-" * 40)
        try:
            markdown_parser = MarkdownNodeParser()
            
            markdown_doc = Document(text=self.sample_texts["technical_content"])
            markdown_nodes = markdown_parser.get_nodes_from_documents([markdown_doc])
            print(f"Number of nodes: {len(markdown_nodes)}")
            for i, node in enumerate(markdown_nodes[:3]):
                print(f"\nNode {i+1} (length: {len(node.text)}):")
                print(textwrap.fill(node.text.strip(), width=80))
        except Exception as e:
            print(f"MarkdownNodeParser not available: {e}")
        
        # 6. SemanticSplitterNodeParser (if embeddings available)
        if self.embedding_model:
            print("\n\n6. SemanticSplitterNodeParser")
            print("-" * 40)
            try:
                semantic_splitter = SemanticSplitterNodeParser(
                    buffer_size=1,
                    breakpoint_percentile_threshold=95,
                    embed_model=self.embedding_model
                )
                
                semantic_nodes = semantic_splitter.get_nodes_from_documents([doc])
                print(f"Number of nodes: {len(semantic_nodes)}")
                for i, node in enumerate(semantic_nodes[:3]):
                    print(f"\nNode {i+1} (length: {len(node.text)}):")
                    print(textwrap.fill(node.text.strip(), width=80))
            except Exception as e:
                print(f"Error with semantic splitter: {e}")
    
    def compare_splitters(self):
        """Compare results from different splitters"""
        print("\n" + "="*60)
        print("SPLITTER COMPARISON ANALYSIS")
        print("="*60)
        
        test_text = self.sample_texts["business_document"]
        
        # Test different splitters
        splitters = {
            "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter(
                chunk_size=300, chunk_overlap=50
            ),
            "CharacterTextSplitter": CharacterTextSplitter(
                chunk_size=300, chunk_overlap=50, separator="\n\n"
            ),
            "TokenTextSplitter": TokenTextSplitter(
                chunk_size=100, chunk_overlap=20
            ),
            "SentenceSplitter": SentenceSplitter(
                chunk_size=300, chunk_overlap=50
            )
        }
        
        print("\nComparison Results:")
        print("-" * 40)
        
        for name, splitter in splitters.items():
            if "Sentence" in name:
                # LlamaIndex splitter
                doc = Document(text=test_text)
                chunks = splitter.get_nodes_from_documents([doc])
                chunk_lengths = [len(node.text) for node in chunks]
                chunk_texts = [node.text for node in chunks]
            else:
                # LangChain splitter
                chunk_texts = splitter.split_text(test_text)
                chunk_lengths = [len(chunk) for chunk in chunk_texts]
            
            print(f"\n{name}:")
            print(f"  - Number of chunks: {len(chunk_texts)}")
            print(f"  - Average chunk length: {sum(chunk_lengths)/len(chunk_lengths):.1f}")
            print(f"  - Min/Max chunk length: {min(chunk_lengths)}/{max(chunk_lengths)}")
            print(f"  - Total characters: {sum(chunk_lengths)}")
    
    def show_chunking_strategies(self):
        """Show different chunking strategies and their use cases"""
        print("\n" + "="*60)
        print("CHUNKING STRATEGIES & USE CASES")
        print("="*60)
        
        strategies = {
            "RecursiveCharacterTextSplitter": {
                "description": "Splits text hierarchically using multiple separators",
                "use_cases": ["General text processing", "Mixed content", "Default choice"],
                "pros": ["Flexible", "Preserves structure", "Good for most texts"],
                "cons": ["Can be slower", "May not respect semantic boundaries"]
            },
            "CharacterTextSplitter": {
                "description": "Splits text using a single separator",
                "use_cases": ["Simple text", "Known separator patterns", "Fast processing"],
                "pros": ["Simple", "Fast", "Predictable"],
                "cons": ["Less flexible", "May create uneven chunks"]
            },
            "SentenceSplitter": {
                "description": "Splits text at sentence boundaries",
                "use_cases": ["Natural language", "Preserving meaning", "Academic texts"],
                "pros": ["Semantic awareness", "Natural boundaries", "Good for QA"],
                "cons": ["Language dependent", "May create small chunks"]
            },
            "TokenTextSplitter": {
                "description": "Splits text based on token count",
                "use_cases": ["LLM input preparation", "Token-limited APIs", "Consistent sizing"],
                "pros": ["Accurate token counting", "LLM-friendly", "Consistent"],
                "cons": ["Requires tokenizer", "May break words"]
            },
            "SemanticSplitter": {
                "description": "Splits text based on semantic similarity",
                "use_cases": ["Coherent topics", "Knowledge base", "RAG applications"],
                "pros": ["Semantic coherence", "Topic preservation", "Smart boundaries"],
                "cons": ["Requires embeddings", "Slower", "More complex"]
            },
            "CodeSplitter": {
                "description": "Splits code while preserving structure",
                "use_cases": ["Source code", "Documentation", "Technical content"],
                "pros": ["Syntax aware", "Preserves functions", "Code-specific"],
                "cons": ["Language specific", "Limited to code"]
            }
        }
        
        for name, info in strategies.items():
            print(f"\n{name}:")
            print(f"  Description: {info['description']}")
            print(f"  Use Cases: {', '.join(info['use_cases'])}")
            print(f"  Pros: {', '.join(info['pros'])}")
            print(f"  Cons: {', '.join(info['cons'])}")

def run_text_splitters_demo():
    """Run the complete text splitters demonstration"""
    print("\n" + "="*80)
    print("TEXT SPLITTERS DEMO: LANGCHAIN VS LLAMAINDEX")
    print("="*80)
    
    # Initialize demo
    demo = TextSplitterDemo()
    
    # Run LangChain splitters demo
    demo.demo_langchain_splitters()
    
    # Run LlamaIndex splitters demo
    demo.demo_llamaindex_splitters()
    
    # Compare splitters
    demo.compare_splitters()
    
    # Show chunking strategies
    demo.show_chunking_strategies()
    
    print("\n" + "="*80)
    print("TEXT SPLITTERS DEMO COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    run_text_splitters_demo()