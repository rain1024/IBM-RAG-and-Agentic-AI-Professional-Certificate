"""
Simple RAG Agent with Vietnamese Wikipedia Data
This script builds a Retrieval-Augmented Generation (RAG) system using Vietnamese Wikipedia articles
about major Vietnamese cities (Hanoi, Ho Chi Minh City, Hai Phong).

Features:
- Document loading from HTML files
- Text chunking and embedding
- Vector storage with FAISS
- Question answering with Azure OpenAI
- Vietnamese language support
"""

import os
import glob
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# HTML processing
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

class VietnameseRAGAgent:
    def __init__(self):
        """Initialize the RAG agent with Azure OpenAI configurations."""
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.documents = []
        
    def setup_azure_openai(self):
        """Set up Azure OpenAI client and embeddings."""
        try:
            # Initialize the language model
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                temperature=0.1,  # Lower temperature for more factual responses
                max_tokens=1000
            )
            
            # Initialize embeddings
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
                azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", 
                                         os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
            )
            
            print("✓ Azure OpenAI services initialized successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error setting up Azure OpenAI: {e}")
            print("Please check your environment variables:")
            print("- AZURE_OPENAI_ENDPOINT")
            print("- AZURE_OPENAI_API_KEY") 
            print("- AZURE_OPENAI_API_VERSION")
            print("- AZURE_OPENAI_DEPLOYMENT_NAME")
            print("- AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME (optional)")
            return False
    
    def load_documents(self, data_folder="script04-data"):
        """Load Vietnamese Wikipedia HTML documents."""
        print(f"\n📖 Loading documents from {data_folder}/...")
        
        # Get HTML files
        html_files = glob.glob(f"{data_folder}/*.html")
        
        if not html_files:
            print(f"✗ No HTML files found in {data_folder}/")
            return False
        
        documents = []
        for file_path in html_files:
            try:
                print(f"  Loading: {os.path.basename(file_path)}")
                
                # Load HTML with BeautifulSoup for better processing
                with open(file_path, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file.read(), 'html.parser')
                
                # Extract main content (remove navigation, scripts, styles)
                for script in soup(["script", "style", "nav", "header", "footer"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                # Create document with metadata
                city_name = self._extract_city_name(os.path.basename(file_path))
                from langchain_core.documents import Document
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "city": city_name,
                        "language": "vietnamese"
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                print(f"  ✗ Error loading {file_path}: {e}")
        
        self.documents = documents
        print(f"✓ Successfully loaded {len(documents)} documents")
        return len(documents) > 0
    
    def _extract_city_name(self, filename):
        """Extract city name from filename."""
        if "Hà Nội" in filename:
            return "Hà Nội"
        elif "Hồ Chí Minh" in filename:
            return "Thành phố Hồ Chí Minh"
        elif "Hải Phòng" in filename:
            return "Hải Phòng"
        else:
            return filename.replace(".html", "")
    
    def process_documents(self):
        """Split documents into chunks and create embeddings."""
        if not self.documents:
            print("✗ No documents to process")
            return False
        
        print("\n🔧 Processing documents...")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=0,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        splits = text_splitter.split_documents(self.documents)[:10]
        print(f"✓ Split documents into {len(splits)} chunks")
        
        # Create vector store with embeddings
        try:
            print("🔮 Creating embeddings and vector store...")
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            print("✓ Vector store created successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error creating vector store: {e}")
            return False
    
    def setup_qa_chain(self):
        """Set up the question-answering chain."""
        if not self.vectorstore:
            print("✗ Vector store not available")
            return False
        
        print("\n⚙️ Setting up QA chain...")
        
        # Create custom prompt template for Vietnamese content
        template = """Bạn là một trợ lý AI chuyên về thông tin các thành phố Việt Nam. 
Sử dụng thông tin được cung cấp để trả lời câu hỏi một cách chính xác và chi tiết.

Ngữ cảnh: {context}

Câu hỏi: {question}

Hãy trả lời bằng tiếng Việt một cách rõ ràng và chính xác. Nếu không tìm thấy thông tin, hãy nói rằng bạn không biết.

Trả lời:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        print("✓ QA chain ready")
        return True
    
    def ask_question(self, question):
        """Ask a question and get an answer."""
        if not self.qa_chain:
            return "❌ QA chain not initialized. Please setup the agent first."
        
        try:
            print(f"\n❓ Question: {question}")
            print("🤔 Thinking...")
            
            result = self.qa_chain.invoke({"query": question})
            answer = result["result"]
            sources = result["source_documents"]
            
            print(f"\n💡 Answer: {answer}")
            
            # Show sources
            if sources:
                print(f"\n📚 Sources ({len(sources)} documents):")
                for i, doc in enumerate(sources[:2], 1):
                    city = doc.metadata.get("city", "Unknown")
                    preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    print(f"  {i}. {city}: {preview}")
            
            return answer
            
        except Exception as e:
            error_msg = f"❌ Error processing question: {e}"
            print(error_msg)
            return error_msg
    
    def initialize(self):
        """Initialize the complete RAG system."""
        print("🚀 Initializing Vietnamese Cities RAG Agent...")
        
        # Step 1: Setup Azure OpenAI
        if not self.setup_azure_openai():
            return False
        
        # Step 2: Load documents
        if not self.load_documents():
            return False
        
        # Step 3: Process documents
        if not self.process_documents():
            return False
        
        # Step 4: Setup QA chain
        if not self.setup_qa_chain():
            return False
        
        print("\n🎉 RAG Agent initialized successfully!")
        print("You can now ask questions about Vietnamese cities.")
        return True

def main():
    """Main function to demonstrate the RAG agent."""
    # Create and initialize the RAG agent
    agent = VietnameseRAGAgent()
    
    if not agent.initialize():
        print("Failed to initialize RAG agent")
        return
    
    print("\n" + "="*60)
    print("🏙️  VIETNAMESE CITIES RAG AGENT")
    print("="*60)
    print("Ask questions about Vietnamese cities (Hanoi, Ho Chi Minh City, Hai Phong)")
    print("Type 'quit' to exit")
    print("="*60)
    
    # Sample questions to try
    sample_questions = [
        "Hà Nội có dân số bao nhiêu?",
        "Thành phố Hồ Chí Minh nằm ở đâu?",
        "Hải Phòng có những đặc điểm gì nổi bật?",
        "So sánh diện tích của các thành phố lớn Việt Nam",
        "Lịch sử của Hà Nội như thế nào?"
    ]
    
    print("\n🎯 Sample questions you can try:")
    for i, q in enumerate(sample_questions, 1):
        print(f"  {i}. {q}")
    
    # Interactive Q&A loop
    while True:
        try:
            print("\n" + "-"*40)
            question = input("\n🔍 Your question (or 'quit'): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                print("Please enter a question.")
                continue
            
            # Get answer
            agent.ask_question(question)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

