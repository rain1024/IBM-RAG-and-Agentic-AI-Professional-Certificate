"""
LangChain Integration with LlamaIndex - LangChainNodeParser Demo
================================================================================

Module: script02.py  
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates the integration of LangChain text splitters with LlamaIndex
    through the LangChainNodeParser. It showcases how to use various LangChain text
    splitters (CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter)
    within a LlamaIndex RAG pipeline for company policies Q&A system.
    
    The demo highlights the interoperability between LangChain and LlamaIndex,
    allowing users to leverage the best text splitting strategies from both ecosystems.
"""

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.node_parser import LangchainNodeParser
from typing import List, Dict, Any
import time

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

class LangChainNodeParserDemo:
    """Demo class for LangChainNodeParser integration with various LangChain text splitters"""
    
    def __init__(self):
        self.documents = []
        self.indexes = {}
        self.query_engines = {}
        
    def create_sample_documents(self) -> List[Document]:
        """Create comprehensive sample documents for text splitting comparison"""
        documents = [
            Document(
                text="""
                COMPANY HANDBOOK - REMOTE WORK POLICY
                
                Overview:
                Our company recognizes the importance of flexible work arrangements to support employee productivity and work-life balance. This comprehensive remote work policy outlines the guidelines, expectations, and requirements for employees working from home or other remote locations.
                
                Eligibility:
                Full-time employees who have completed their probationary period (typically 6 months) are eligible to request remote work arrangements. Part-time employees may also be considered on a case-by-case basis. All remote work arrangements must be approved by the direct supervisor and HR department.
                
                Application Process:
                1. Submit formal request using Form RW-001 at least 30 days in advance
                2. Include detailed work plan and productivity metrics
                3. Demonstrate reliable internet connectivity (minimum 25 Mbps download/upload)
                4. Provide evidence of dedicated workspace meeting ergonomic standards
                5. Agree to regular check-ins and performance monitoring
                
                Technical Requirements:
                - High-speed internet connection (minimum 25 Mbps)
                - Company-provided laptop or desktop computer
                - Webcam and microphone for video conferencing
                - Access to company VPN and security software
                - Secure file storage and backup systems
                
                Communication Expectations:
                Remote workers must maintain regular communication with their team and supervisor. This includes:
                - Daily check-ins via Slack or Microsoft Teams
                - Weekly one-on-one meetings with supervisor
                - Monthly team meetings (virtual or in-person)
                - Quarterly performance reviews
                - Immediate response to urgent communications within 2 hours during business hours
                
                Performance Standards:
                Remote workers are expected to maintain the same level of productivity and quality as office-based employees. Key performance indicators include:
                - Meeting all project deadlines and deliverables
                - Maintaining consistent work schedule during core hours (9 AM - 3 PM)
                - Achieving quarterly performance goals
                - Participating actively in team meetings and collaborative sessions
                
                Security and Confidentiality:
                All remote workers must adhere to company security protocols:
                - Use only company-approved software and applications
                - Maintain confidentiality of sensitive information
                - Report security incidents immediately
                - Complete monthly security training modules
                - Ensure physical security of work environment
                
                Equipment and Expenses:
                The company will provide necessary equipment including laptop, monitor, keyboard, mouse, and office chair. Employees are responsible for:
                - Internet connectivity costs
                - Utilities and office space setup
                - Maintenance and care of company equipment
                - Return of all equipment upon employment termination
                
                Review and Modification:
                This policy will be reviewed annually and may be modified based on business needs, employee feedback, and industry best practices. All changes will be communicated to employees at least 30 days in advance.
                """,
                metadata={"source": "company_handbook", "section": "remote_work_policy", "last_updated": "2024-01-15"}
            ),
            Document(
                text="""
                EMPLOYEE BENEFITS COMPREHENSIVE GUIDE
                
                Introduction:
                We are committed to providing our employees with a comprehensive benefits package that supports their health, financial security, and professional development. This guide outlines all available benefits and how to access them.
                
                HEALTH INSURANCE
                
                Medical Coverage:
                Our health insurance plan covers 100% of premium costs for employees and 80% for dependents. The plan includes:
                - Preventive care (annual checkups, vaccinations)
                - Emergency room visits and urgent care
                - Prescription drug coverage
                - Mental health and substance abuse treatment
                - Maternity and newborn care
                - Specialist consultations and procedures
                
                Dental Insurance:
                Complete dental coverage including:
                - Routine cleanings and checkups (100% coverage)
                - Fillings and basic procedures (80% coverage)
                - Major procedures like crowns and bridges (60% coverage)
                - Orthodontic treatment (50% coverage up to $2,000 lifetime maximum)
                
                Vision Insurance:
                Comprehensive vision care covering:
                - Annual eye exams
                - Prescription glasses or contact lenses
                - Frames allowance up to $200 annually
                - Discounts on LASIK and other corrective surgeries
                
                RETIREMENT PLANS
                
                401(k) Plan:
                - Company matches 50% of contributions up to 6% of salary
                - Immediate vesting for employee contributions
                - Graded vesting for employer contributions (20% per year)
                - Access to financial planning resources and investment advice
                - Loan options available for financial emergencies
                
                Pension Plan:
                For employees hired before 2020, traditional pension plan provides:
                - Monthly retirement income based on salary and years of service
                - Full vesting after 5 years of service
                - Early retirement options with reduced benefits
                - Survivor benefits for eligible dependents
                
                PAID TIME OFF
                
                Vacation Leave:
                - 0-2 years: 15 days annually
                - 3-5 years: 20 days annually
                - 6-10 years: 25 days annually
                - 10+ years: 30 days annually
                - Maximum carryover: 5 days to following year
                
                Sick Leave:
                - 12 days annually for all employees
                - Medical certification required for absences exceeding 3 consecutive days
                - Can be used for personal illness or caring for immediate family members
                - Unused days do not carry over to next year
                
                Personal Days:
                - 3 personal days annually for all employees
                - Can be used for any purpose without explanation
                - Must be scheduled in advance except for emergencies
                - Cannot be carried over to following year
                
                PROFESSIONAL DEVELOPMENT
                
                Training Budget:
                - $2,000 annually per employee for professional development
                - Covers conferences, workshops, online courses, and certifications
                - Reimbursement requires pre-approval and completion certificates
                - Unused budget does not carry over to next year
                
                Tuition Reimbursement:
                - Up to $5,000 annually for job-related degree programs
                - Requires maintaining minimum GPA of 3.0
                - Employee must commit to staying with company for 2 years after graduation
                - Prorated repayment required if leaving before commitment period
                
                ADDITIONAL BENEFITS
                
                Life Insurance:
                - Basic life insurance equal to 2x annual salary provided at no cost
                - Optional additional coverage available for purchase
                - Accidental death and dismemberment coverage included
                - Portable coverage available upon employment termination
                
                Flexible Spending Accounts:
                - Healthcare FSA: up to $2,850 annually
                - Dependent Care FSA: up to $5,000 annually
                - Commuter benefits for public transportation and parking
                - Use-it-or-lose-it policy with 2.5 month grace period
                
                Employee Assistance Program:
                - Confidential counseling services for personal and work-related issues
                - Legal consultation and financial planning services
                - Work-life balance resources and referrals
                - Crisis intervention and emergency support
                - Available 24/7 via phone or online portal
                """,
                metadata={"source": "company_handbook", "section": "employee_benefits", "last_updated": "2024-02-01"}
            ),
            Document(
                text="""
                PERFORMANCE MANAGEMENT SYSTEM GUIDE
                
                Philosophy:
                Our performance management system is designed to foster employee growth, recognize achievements, and align individual goals with organizational objectives. We believe in continuous feedback, regular coaching, and fair evaluation processes.
                
                PERFORMANCE REVIEW CYCLE
                
                Annual Reviews:
                Comprehensive annual performance reviews are conducted in Q4 for all employees. The process includes:
                - Self-assessment completion by employee
                - 360-degree feedback from peers, subordinates, and supervisors
                - Goal achievement evaluation against established metrics
                - Competency assessment in key areas
                - Development planning for the following year
                - Salary and promotion considerations
                
                Mid-Year Check-ins:
                Mandatory mid-year reviews occur in Q2 to:
                - Assess progress toward annual goals
                - Identify obstacles and provide support
                - Adjust goals if necessary due to changing business needs
                - Provide feedback on performance trends
                - Discuss career development opportunities
                
                Quarterly One-on-Ones:
                Monthly one-on-one meetings between employees and supervisors focus on:
                - Project updates and current challenges
                - Short-term goal setting and prioritization
                - Feedback on recent accomplishments
                - Career development discussions
                - Resource needs and support requirements
                
                GOAL SETTING FRAMEWORK
                
                SMART Goals:
                All employee goals must be:
                - Specific: Clearly defined and unambiguous
                - Measurable: Quantifiable with concrete metrics
                - Achievable: Realistic given available resources and constraints
                - Relevant: Aligned with role responsibilities and company objectives
                - Time-bound: Has definite start and end dates
                
                Goal Categories:
                1. Performance Goals (60% weight)
                   - Primary job responsibilities and key deliverables
                   - Quality and efficiency metrics
                   - Customer satisfaction and service levels
                
                2. Development Goals (25% weight)
                   - Skill enhancement and learning objectives
                   - Professional certifications and training
                   - Cross-functional collaboration and knowledge sharing
                
                3. Behavioral Goals (15% weight)
                   - Demonstration of company values
                   - Leadership and teamwork behaviors
                   - Innovation and continuous improvement initiatives
                
                PERFORMANCE RATINGS
                
                Rating Scale:
                - Outstanding (5): Consistently exceeds expectations in all areas
                - Exceeds Expectations (4): Frequently surpasses goals and standards
                - Meets Expectations (3): Consistently achieves goals and standards
                - Below Expectations (2): Occasionally falls short of expectations
                - Unsatisfactory (1): Consistently fails to meet minimum requirements
                
                Rating Distribution:
                - Outstanding: 10-15% of employees
                - Exceeds Expectations: 20-25% of employees
                - Meets Expectations: 50-60% of employees
                - Below Expectations: 10-15% of employees
                - Unsatisfactory: 0-5% of employees
                
                COMPENSATION DECISIONS
                
                Merit Increases:
                Based on performance ratings and market analysis:
                - Outstanding: 5-7% salary increase
                - Exceeds Expectations: 3-5% salary increase
                - Meets Expectations: 2-3% salary increase
                - Below Expectations: 0-2% salary increase
                - Unsatisfactory: No increase, possible corrective action
                
                Bonus Eligibility:
                - Outstanding: 15-20% of annual salary
                - Exceeds Expectations: 10-15% of annual salary
                - Meets Expectations: 5-10% of annual salary
                - Below Expectations: 0-5% of annual salary
                - Unsatisfactory: No bonus eligibility
                
                DEVELOPMENT PLANNING
                
                Career Pathing:
                - Individual development plans created for each employee
                - Clear progression paths within current role and department
                - Cross-functional opportunities and stretch assignments
                - Mentorship programs and leadership development tracks
                - Regular career counseling and guidance sessions
                
                Skill Assessment:
                - Technical competencies mapped to role requirements
                - Soft skills evaluation and development recommendations
                - Training needs analysis and prioritization
                - External development opportunities and resources
                - Succession planning for key positions
                
                PERFORMANCE IMPROVEMENT
                
                Performance Improvement Plans (PIPs):
                For employees rated Below Expectations or Unsatisfactory:
                - 90-day improvement plan with specific milestones
                - Weekly check-ins with supervisor and HR
                - Additional training and support resources
                - Clear consequences for failure to improve
                - Documentation of all interventions and progress
                
                Success Metrics:
                - 70% of employees on PIPs successfully improve to Meets Expectations
                - Regular monitoring and adjustment of improvement strategies
                - Post-PIP follow-up for sustained performance
                - Recognition and celebration of improvement achievements
                """,
                metadata={"source": "company_handbook", "section": "performance_management", "last_updated": "2024-01-25"}
            )
        ]
        return documents
    
    def setup_langchain_splitters(self):
        """Setup different LangChain text splitters for comparison"""
        try:
            # Import LangChain text splitters
            from langchain.text_splitter import (
                CharacterTextSplitter,
                RecursiveCharacterTextSplitter,
                TokenTextSplitter
            )
            
            # Character-based splitter
            char_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
            )
            
            # Recursive character splitter (more intelligent)
            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Token-based splitter
            token_splitter = TokenTextSplitter(
                chunk_size=256,
                chunk_overlap=50,
            )
            
            # Create LangChainNodeParser instances
            langchain_parsers = {
                "character": LangchainNodeParser(lc_splitter=char_splitter),
                "recursive": LangchainNodeParser(lc_splitter=recursive_splitter),
                "token": LangchainNodeParser(lc_splitter=token_splitter)
            }
            
            return langchain_parsers
            
        except ImportError as e:
            print(f"Error importing LangChain: {e}")
            print("Please install langchain: pip install langchain")
            return {}
    
    def create_indexes_with_different_splitters(self):
        """Create vector indexes using different LangChain text splitters"""
        print("\n=== Creating Indexes with Different LangChain Text Splitters ===")
        
        # Get sample documents
        self.documents = self.create_sample_documents()
        
        # Setup LangChain splitters
        langchain_parsers = self.setup_langchain_splitters()
        
        if not langchain_parsers:
            print("Could not setup LangChain parsers. Exiting.")
            return
        
        # Create indexes for each splitter
        for splitter_name, parser in langchain_parsers.items():
            print(f"\n--- Processing with {splitter_name.upper()} splitter ---")
            
            start_time = time.time()
            
            # Parse documents into nodes
            nodes = parser.get_nodes_from_documents(self.documents)
            
            # Create vector index
            index = VectorStoreIndex(nodes, show_progress=True)
            
            # Store index and create query engine
            self.indexes[splitter_name] = index
            self.query_engines[splitter_name] = index.as_query_engine(
                similarity_top_k=3,
                response_mode="compact"
            )
            
            end_time = time.time()
            
            print(f"✓ {splitter_name.upper()} splitter completed")
            print(f"  - Nodes created: {len(nodes)}")
            print(f"  - Processing time: {end_time - start_time:.2f}s")
            
            # Show sample node content
            if nodes:
                print(f"  - Sample node preview:")
                print(f"    Text length: {len(nodes[0].text)} chars")
                print(f"    Text preview: {nodes[0].text[:150]}...")
                print(f"    Metadata: {nodes[0].metadata}")
    
    def compare_splitter_performance(self):
        """Compare performance of different text splitters"""
        print("\n=== Comparing Text Splitter Performance ===")
        
        test_queries = [
            "What is the remote work policy?",
            "How does the performance review process work?",
            "What are the health insurance benefits?",
            "What is the professional development budget?",
            "How are performance ratings calculated?"
        ]
        
        results = {}
        
        for query in test_queries:
            print(f"\n--- Query: '{query}' ---")
            results[query] = {}
            
            for splitter_name, query_engine in self.query_engines.items():
                print(f"\n{splitter_name.upper()} Splitter:")
                
                start_time = time.time()
                response = query_engine.query(query)
                end_time = time.time()
                
                results[query][splitter_name] = {
                    "response": str(response),
                    "response_time": end_time - start_time,
                    "source_nodes": len(response.source_nodes) if hasattr(response, 'source_nodes') else 0
                }
                
                print(f"  Response time: {end_time - start_time:.2f}s")
                print(f"  Response: {str(response)[:200]}...")
                if hasattr(response, 'source_nodes'):
                    print(f"  Source nodes: {len(response.source_nodes)}")
        
        return results
    
    def analyze_node_characteristics(self):
        """Analyze characteristics of nodes created by different splitters"""
        print("\n=== Analyzing Node Characteristics ===")
        
        for splitter_name, index in self.indexes.items():
            print(f"\n--- {splitter_name.upper()} Splitter Analysis ---")
            
            # Get all nodes from the index
            nodes = []
            for doc_id in index.docstore.docs:
                node = index.docstore.get_node(doc_id)
                nodes.append(node)
            
            if nodes:
                # Calculate statistics
                node_lengths = [len(node.text) for node in nodes]
                avg_length = sum(node_lengths) / len(node_lengths)
                min_length = min(node_lengths)
                max_length = max(node_lengths)
                
                print(f"  Total nodes: {len(nodes)}")
                print(f"  Average node length: {avg_length:.0f} characters")
                print(f"  Min node length: {min_length} characters")
                print(f"  Max node length: {max_length} characters")
                print(f"  Length distribution:")
                
                # Length distribution
                length_buckets = {"<500": 0, "500-1000": 0, "1000-1500": 0, ">1500": 0}
                for length in node_lengths:
                    if length < 500:
                        length_buckets["<500"] += 1
                    elif length < 1000:
                        length_buckets["500-1000"] += 1
                    elif length < 1500:
                        length_buckets["1000-1500"] += 1
                    else:
                        length_buckets[">1500"] += 1
                
                for bucket, count in length_buckets.items():
                    percentage = (count / len(nodes)) * 100
                    print(f"    {bucket} chars: {count} nodes ({percentage:.1f}%)")
    
    def demonstrate_langchain_integration(self):
        """Demonstrate the integration features of LangChainNodeParser"""
        print("\n=== Demonstrating LangChain Integration Features ===")
        
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            # Create a custom splitter with specific parameters
            custom_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
                keep_separator=True
            )
            
            # Wrap with LangChainNodeParser
            langchain_parser = LangchainNodeParser(
                lc_splitter=custom_splitter,
                include_metadata=True,
                include_prev_next_rel=True
            )
            
            print("✓ Custom LangChain splitter created with parameters:")
            print(f"  - Chunk size: {custom_splitter._chunk_size}")
            print(f"  - Chunk overlap: {custom_splitter._chunk_overlap}")
            print(f"  - Separators: {custom_splitter._separators}")
            print(f"  - Keep separator: {custom_splitter._keep_separator}")
            
            # Process documents
            nodes = langchain_parser.get_nodes_from_documents(self.documents[:1])  # Use first document
            
            print(f"\n✓ Processed {len(nodes)} nodes")
            
            # Show node relationships
            if len(nodes) > 1:
                print(f"\n✓ Node relationships preserved:")
                for i, node in enumerate(nodes[:3]):  # Show first 3 nodes
                    print(f"  Node {i}:")
                    print(f"    ID: {node.node_id}")
                    print(f"    Text length: {len(node.text)}")
                    print(f"    Relationships: {list(node.relationships.keys())}")
                    if hasattr(node, 'metadata'):
                        print(f"    Metadata keys: {list(node.metadata.keys())}")
            
            return nodes
            
        except ImportError as e:
            print(f"Error: {e}")
            return []

def run_langchain_nodeparser_demo():
    """Run the comprehensive LangChainNodeParser demonstration"""
    print("\n" + "="*80)
    print("LangChain Integration with LlamaIndex - LangChainNodeParser Demo")
    print("="*80)
    
    # Initialize demo
    demo = LangChainNodeParserDemo()
    
    # Create indexes with different splitters
    demo.create_indexes_with_different_splitters()
    
    if not demo.indexes:
        print("❌ Could not create indexes. Please check LangChain installation.")
        return
    
    # Analyze node characteristics
    demo.analyze_node_characteristics()
    
    # Compare performance
    performance_results = demo.compare_splitter_performance()
    
    # Demonstrate integration features
    demo.demonstrate_langchain_integration()
    
    print("\n" + "="*80)
    print("✅ LangChain-LlamaIndex Integration Demo Completed!")
    print("="*80)
    
    # Summary
    print("\n📊 SUMMARY:")
    print("• Successfully integrated LangChain text splitters with LlamaIndex")
    print("• Compared different splitting strategies (Character, Recursive, Token)")
    print("• Analyzed node characteristics and performance metrics")
    print("• Demonstrated preserving relationships and metadata")
    print("• Showed custom splitter configuration options")
    
    print("\n🔧 Key Benefits of LangChainNodeParser:")
    print("• Leverage LangChain's mature text splitting algorithms")
    print("• Maintain LlamaIndex's node structure and relationships")
    print("• Flexible configuration for different use cases")
    print("• Seamless integration between two powerful frameworks")
    print("• Best of both worlds: LangChain splitting + LlamaIndex indexing")

if __name__ == "__main__":
    run_langchain_nodeparser_demo()

