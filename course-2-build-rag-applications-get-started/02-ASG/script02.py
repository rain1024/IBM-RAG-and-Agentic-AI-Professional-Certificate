"""
Sentiment Analysis Gradio Application using LLM
================================================================================

Module: script02.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates a sentiment analysis application using Azure OpenAI LLM
    and Gradio. It provides sentiment analysis capabilities through an easy-to-use
    web interface, analyzing text sentiment and providing detailed insights.
"""

import gradio as gr
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()

# Configure Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.7,
    max_tokens=1000
)

def analyze_sentiment(input_text, analysis_type):
    """
    Analyze sentiment of the input text using Azure OpenAI LLM
    """
    if not input_text.strip():
        return "Please enter some text to analyze."
    
    try:
        if analysis_type == "Basic Sentiment":
            prompt = f"""
            Analyze the sentiment of the following text and classify it as Positive, Negative, or Neutral.
            Provide a brief explanation for your classification.
            
            Text: "{input_text}"
            
            Response format:
            Sentiment: [Positive/Negative/Neutral]
            Confidence: [High/Medium/Low]
            Explanation: [Brief explanation]
            """
            
        elif analysis_type == "Detailed Analysis":
            prompt = f"""
            Perform a detailed sentiment analysis of the following text. Include:
            1. Overall sentiment (Positive/Negative/Neutral)
            2. Confidence level
            3. Emotional tone analysis
            4. Key phrases that influenced the sentiment
            5. Sentiment score (1-10 scale)
            
            Text: "{input_text}"
            """
            
        elif analysis_type == "Emotion Detection":
            prompt = f"""
            Analyze the emotions present in the following text. Identify:
            1. Primary emotion
            2. Secondary emotions (if any)
            3. Emotional intensity (1-10 scale)
            4. Contextual factors affecting emotion
            
            Text: "{input_text}"
            """
            
        elif analysis_type == "Business Impact":
            prompt = f"""
            Analyze this text from a business perspective:
            1. Customer sentiment (Positive/Negative/Neutral)
            2. Urgency level (High/Medium/Low)
            3. Business impact assessment
            4. Recommended action
            
            Text: "{input_text}"
            """
            
        else:
            return "Unknown analysis type selected."
        
        # Send prompt to LLM
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}\n\nPlease check your Azure OpenAI configuration."

def create_gradio_interface():
    """
    Create and configure the Gradio interface for sentiment analysis
    """
    demo = gr.Interface(
        fn=analyze_sentiment,
        inputs=[
            gr.Textbox(
                label="Input Text",
                placeholder="Enter your text for sentiment analysis...",
                lines=5,
                max_lines=10
            ),
            gr.Dropdown(
                label="Analysis Type",
                choices=["Basic Sentiment", "Detailed Analysis", "Emotion Detection", "Business Impact"],
                value="Basic Sentiment"
            )
        ],
        outputs=gr.Textbox(
            label="Sentiment Analysis Results",
            lines=8,
            max_lines=15
        ),
        title="AI-Powered Sentiment Analysis",
        description="Analyze the sentiment of your text using advanced AI. Choose different analysis types for various insights.",
        theme=gr.themes.Soft(),
        examples=[
            ["I absolutely love this product! It's amazing and works perfectly.", "Basic Sentiment"],
            ["The service was okay, nothing special but not terrible either.", "Detailed Analysis"],
            ["I'm so frustrated with this delay! This is completely unacceptable.", "Emotion Detection"],
            ["Our customers are complaining about the new update. Many are requesting refunds.", "Business Impact"],
        ],
        allow_flagging="never"
    )
    
    return demo

if __name__ == "__main__":
    # Create and launch the Gradio interface
    demo = create_gradio_interface()
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
