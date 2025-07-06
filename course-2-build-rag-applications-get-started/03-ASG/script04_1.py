from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter
)
import textwrap


from llama_index.core.node_parser import (
    SentenceSplitter,
    TokenTextSplitter as LlamaTokenTextSplitter,
    CodeSplitter,
    MarkdownNodeParser,
    SimpleNodeParser
)
from llama_index.core import Document

text = """
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
"""

# ================================ #
# Langchain                        #
# ================================ #

print("="*40)
print("Langchain Splitter")
print("="*40)

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=300,
#     chunk_overlap=50,
#     length_function=len,
#     separators=["\n\n", "\n", " ", ""]
# )

splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    separator="\n\n"
)

splitter = MarkdownTextSplitter(
    chunk_size=500,
    chunk_overlap=30
)

print("Text length: ", len(text))
recursive_chunks = splitter.split_text(text)
print(f"Number of chunks: {len(recursive_chunks)}")
for i, chunk in enumerate(recursive_chunks):  # Show first 3 chunks
    print(f"\nChunk {i+1} (length: {len(chunk)}):")
    print(textwrap.fill(chunk.strip(), width=80))


# ================================ #
# Llama-Index                      #
# ================================ #
print("\n" + "="*40)
print("Llama-Index SentenceSplitter")
print("="*40)
splitter = SentenceSplitter(
    chunk_size=100,
    chunk_overlap=10,
    paragraph_separator="\n\n"
)

doc = Document(text=text)
sentence_nodes = splitter.get_nodes_from_documents([doc])
print(f"Number of nodes: {len(sentence_nodes)}")
for i, node in enumerate(sentence_nodes):
    print(f"\nNode {i+1} (length: {len(node.text)}):")
    print(textwrap.fill(node.text.strip(), width=80))

