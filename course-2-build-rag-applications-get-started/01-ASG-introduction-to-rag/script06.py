"""
Neural Networks in RAG Embeddings: Text-to-Vector Transformation Demo
================================================================================

Module: script06.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates the role of neural networks in the embedding step using
    text-embedding-ada-002. Neural networks transform text into high-dimensional
    vector representations (embeddings) that capture semantic meaning and relationships.
    The text-embedding-ada-002 model uses transformer architecture to convert words,
    phrases, and sentences into 1536-dimensional vectors, enabling similarity search
    and semantic understanding for RAG applications.
"""
import os
import numpy as np
import tiktoken
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002"),
    model="text-embedding-ada-002"
)

# Initialize tokenizer for text-embedding-ada-002
tokenizer = tiktoken.get_encoding("cl100k_base")  # Used by text-embedding-ada-002

"""Demo 1: Tokenization and Token IDs"""
print("\n=== Demo 1: Text Tokenization and Token IDs ===")

# Sample mobile company policy texts
sample_text = "Mobile data usage policy: Customers can use up to 50GB of high-speed data per month."

# Tokenize the text
tokens = tokenizer.encode(sample_text)
decoded_tokens = [tokenizer.decode([token]) for token in tokens]

print(f"Original text: {sample_text}")
print(f"Number of tokens: {len(tokens)}")
print(f"Token IDs: {tokens}")
print(f"Decoded tokens: {decoded_tokens}")

# Show token-by-token breakdown
print("\n--- Token Breakdown ---")
for i, (token_id, token_text) in enumerate(zip(tokens, decoded_tokens)):
    print(f"Token {i+1}: ID={token_id:5d} -> '{token_text}'")

"""Demo 2: Generate embeddings for sample texts"""
print("\n=== Demo 2: Text to Vector Embeddings ===")

# Generate embeddings for each text
print("Generating embeddings for sample text...")

embedding = embeddings.embed_query(sample_text)

# Display embedding information
print(f"\nText: {sample_text}")
print(f"Embedding dimensions: {len(embedding)}")
print(f"Embedding type: {type(embedding)}")
print(f"First 5 values: {embedding[:5]}")
print(f"Last 5 values: {embedding[-5:]}")
print(f"Embedding vector magnitude: {np.linalg.norm(embedding):.4f}")
