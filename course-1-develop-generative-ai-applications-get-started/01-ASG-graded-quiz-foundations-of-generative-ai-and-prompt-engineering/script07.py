from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

def setup_azure_openai():
    """
    Set up Azure OpenAI client using environment variables.
    
    Required environment variables:
    - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
    - AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint
    - AZURE_OPENAI_API_VERSION: API version (e.g., "2023-12-01-preview")
    - AZURE_OPENAI_DEPLOYMENT_NAME: Your deployment name
    """
    
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=0.7,
            max_tokens=500
        )
        return llm
    except Exception as e:
        print(f"Error setting up Azure OpenAI: {e}")
        print("Please ensure all required environment variables are set.")
        return None
    
llm = setup_azure_openai()
output_parser = StrOutputParser()

# Dùng toán tử pipe `|` để tạo chuỗi
prompt = ChatPromptTemplate.from_template(
    "Hãy nghĩ ra một slogan hay cho một công ty bán {product}."
)
chain = prompt | llm | output_parser

print(chain.invoke({"product": "đồ thể thao"}))
print(chain.invoke({"product": "công nghệ AI"}))