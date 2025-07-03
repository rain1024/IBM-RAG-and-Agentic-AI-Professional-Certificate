"""
Simple demonstration of PromptTemplate format() method in LangChain
This shows how placeholder variables are replaced with actual values
"""

from langchain.prompts import PromptTemplate

def main():
    print("=== PromptTemplate format() Method Demonstration ===\n")
    
    # Example 1: Simple greeting template
    print("1. Simple Greeting Template:")
    greeting_template = PromptTemplate(
        input_variables=["name", "day"],
        template="Hello {name}! I hope you're having a great {day}."
    )
    
    print(f"Template: {greeting_template.template}")
    print(f"Variables: {greeting_template.input_variables}")
    
    # Using format() to replace placeholders
    formatted_greeting = greeting_template.format(name="Alice", day="Monday")
    print(f"After format(): {formatted_greeting}\n")
    
    # Example 2: Email template
    print("2. Email Template:")
    email_template = PromptTemplate(
        input_variables=["recipient", "subject", "sender"],
        template="""Dear {recipient},

I hope this email finds you well. I wanted to reach out regarding {subject}.

Best regards,
{sender}"""
    )
    
    print(f"Template:\n{email_template.template}")
    print(f"Variables: {email_template.input_variables}")
    
    # Using format() to create actual email
    formatted_email = email_template.format(
        recipient="Dr. Smith",
        subject="the research proposal",
        sender="John Doe"
    )
    print(f"After format():\n{formatted_email}\n")
    
    # Example 3: Question-answering template
    print("3. Question-Answering Template:")
    qa_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""Based on the following context:

{context}

Answer this question: {question}

Answer:"""
    )
    
    print(f"Template:\n{qa_template.template}")
    print(f"Variables: {qa_template.input_variables}")
    
    # Using format() for Q&A
    formatted_qa = qa_template.format(
        context="LangChain is a framework for developing applications powered by language models.",
        question="What is LangChain?"
    )
    print(f"After format():\n{formatted_qa}")
    
    print("\n=== Summary ===")
    print("The format() method's primary function is:")
    print("✓ To replace placeholder variables (like {name}) with actual values")
    print("✓ This creates the final prompt that can be sent to a language model")
    print("✓ Variables are defined in curly braces {} in the template")
    print("✓ The format() method takes keyword arguments matching the variable names")

if __name__ == "__main__":
    main()
