"""
Text Summarization Gradio Application using LLM
================================================================================

Module: script05.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates a text summarization application using Azure OpenAI LLM
    and Gradio. It provides text summarization capabilities through an easy-to-use
    web interface, allowing users to summarize long texts in different languages
    and various summary lengths.
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
    temperature=0.8,
    max_tokens=1000
)

def summarize_text(text, language, summary_length):
    """
    Summarize text using Azure OpenAI LLM
    """
    if not text.strip():
        return "Vui lòng nhập văn bản cần tóm tắt."
    
    try:
        # Determine summary length
        if summary_length == "Ngắn":
            length_instruction = "very brief, 1-2 sentences"
        elif summary_length == "Trung bình":
            length_instruction = "moderate length, 3-5 sentences"
        else:  # Dài
            length_instruction = "detailed, 6-10 sentences"
        
        # Language-specific instructions
        language_instructions = {
            "English": "Summarize the text in English",
            "Vietnamese": "Tóm tắt văn bản bằng tiếng Việt",
            "Chinese": "用中文总结文本"
        }
        
        prompt = f"""
        Please summarize the following text in {length_instruction}.
        
        Requirements:
        - {language_instructions.get(language, "Summarize the text in English")}
        - Capture the main points and key information
        - Make it clear and concise
        - Maintain the original meaning
        - Use proper grammar and structure
        
        Text to summarize:
        {text}
        
        Format your response as:
        📝 Tóm tắt: [Summary content]
        🔍 Điểm chính: [Key points]
        """
        
        # Send prompt to LLM
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        return f"Lỗi khi tóm tắt văn bản: {str(e)}\n\nVui lòng kiểm tra cấu hình Azure OpenAI."

def create_gradio_interface():
    """
    Create and configure the Gradio interface for text summarization
    """
    demo = gr.Interface(
        fn=summarize_text,
        inputs=[
            gr.Textbox(
                label="Văn bản cần tóm tắt",
                placeholder="Nhập văn bản dài cần tóm tắt...",
                lines=10,
                max_lines=20
            ),
            gr.Dropdown(
                label="Ngôn ngữ",
                choices=["English", "Vietnamese", "Chinese"],
                value="Vietnamese"
            ),
            gr.Dropdown(
                label="Độ dài tóm tắt",
                choices=["Ngắn", "Trung bình", "Dài"],
                value="Trung bình"
            )
        ],
        outputs=gr.Textbox(
            label="Tóm tắt văn bản",
            lines=8,
            max_lines=15
        ),
        title="📝 Ứng dụng tóm tắt văn bản bằng AI",
        description="Tóm tắt văn bản dài một cách thông minh bằng trí tuệ nhân tạo. Chọn ngôn ngữ và độ dài tóm tắt phù hợp để nhận được bản tóm tắt chất lượng cao!",
        theme=gr.themes.Soft(),
        examples=[
            ["Trí tuệ nhân tạo (AI) là một lĩnh vực của khoa học máy tính tập trung vào việc tạo ra các máy móc có thể thực hiện các nhiệm vụ thường đòi hỏi trí thông minh của con người. AI bao gồm nhiều kỹ thuật khác nhau như học máy, học sâu, xử lý ngôn ngữ tự nhiên, và thị giác máy tính. Trong những năm gần đây, AI đã có những bước tiến vượt bậc và được ứng dụng rộng rãi trong nhiều lĩnh vực từ y tế, giáo dục đến tài chính và giao thông.", "Vietnamese", "Trung bình"],
            ["Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data without being explicitly programmed. It uses statistical techniques to give computers the ability to progressively improve their performance on a specific task through experience. Common applications include recommendation systems, image recognition, natural language processing, and predictive analytics.", "English", "Ngắn"],
            ["气候变化是指由于人类活动导致的全球或区域气候模式的长期变化。主要原因包括燃烧化石燃料、森林砍伐和工业活动，这些活动增加了大气中的温室气体浓度。气候变化的影响包括全球变暖、海平面上升、极端天气事件增多、生态系统破坏等。", "Chinese", "Dài"],
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
