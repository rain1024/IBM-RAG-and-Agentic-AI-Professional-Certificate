"""
Simple Text Processing Gradio Application
================================================================================

Module: script01.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates a simple Gradio web application for text processing.
    It provides basic text operations like case conversion, word counting, text reversal,
    and simple summarization through an easy-to-use web interface.
"""

import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def process_text(input_text, processing_type):
    """
    Simple text processing function that can be extended for RAG applications
    """
    if not input_text.strip():
        return "Please enter some text to process."
    
    if processing_type == "Upper Case":
        return input_text.upper()
    elif processing_type == "Lower Case":
        return input_text.lower()
    elif processing_type == "Word Count":
        word_count = len(input_text.split())
        return f"Word count: {word_count}\n\nOriginal text:\n{input_text}"
    elif processing_type == "Reverse":
        return input_text[::-1]
    elif processing_type == "Summary":
        # Simple summary (first 100 characters + "...")
        if len(input_text) > 100:
            return input_text[:100] + "..."
        else:
            return input_text
    else:
        return "Unknown processing type selected."

def create_gradio_interface():
    """
    Create and configure the Gradio interface using gr.Interface
    """
    demo = gr.Interface(
        fn=process_text,
        inputs=[
            gr.Textbox(
                label="Input Text",
                placeholder="Enter your text here...",
                lines=5,
                max_lines=10
            ),
            gr.Dropdown(
                label="Processing Type",
                choices=["Upper Case", "Lower Case", "Word Count", "Reverse", "Summary"],
                value="Word Count"
            )
        ],
        outputs=gr.Textbox(
            label="Output",
            lines=5,
            max_lines=10
        ),
        title="Simple RAG Text Processor",
        description="This is a simple Gradio app that can be extended for RAG applications.",
        theme=gr.themes.Soft(),
        examples=[
            ["Hello World! This is a simple text processing example.", "Word Count"],
            ["The quick brown fox jumps over the lazy dog.", "Upper Case"],
            ["LOREM IPSUM DOLOR SIT AMET, CONSECTETUR ADIPISCING ELIT.", "Lower Case"],
            ["This is a longer text that will be summarized when you select the Summary option. It contains multiple sentences and should demonstrate the summary functionality.", "Summary"],
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
