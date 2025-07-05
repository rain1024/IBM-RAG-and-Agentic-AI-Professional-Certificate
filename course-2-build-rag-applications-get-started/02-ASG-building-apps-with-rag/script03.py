"""
Joke Generation Gradio Application using LLM
================================================================================

Module: script03.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates a joke generation application using Azure OpenAI LLM
    and Gradio. It provides joke generation capabilities through an easy-to-use
    web interface, generating jokes based on language, age, and topic preferences.
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

def generate_joke(topic, language, age):
    """
    Generate a joke based on topic, language, and age using Azure OpenAI LLM
    """
    if not topic.strip():
        return "Vui lòng nhập chủ đề cho truyện cười."
    
    try:
        # Create age-appropriate content guidelines
        if age <= 12:
            age_guidance = "suitable for children, innocent and educational"
        elif age <= 17:
            age_guidance = "suitable for teenagers, avoid adult themes"
        else:
            age_guidance = "suitable for adults, can include mature humor"
        
        # Language-specific instructions
        language_instructions = {
            "English": "Generate the joke in English",
            "Vietnamese": "Tạo truyện cười bằng tiếng Việt",
            "Chinese": "用中文生成笑话"
        }
        
        prompt = f"""
        Generate a funny, clean joke about "{topic}" that is {age_guidance}.
        
        Requirements:
        - {language_instructions.get(language, "Generate the joke in English")}
        - Age-appropriate for {age} years old
        - Topic: {topic}
        - Make it genuinely funny and creative
        - Keep it appropriate and respectful
        - Length: 2-5 sentences
        
        Format your response as:
        🎭 Joke: [The joke content]
        😄 Why it's funny: [Brief explanation]
        """
        
        # Send prompt to LLM
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        return f"Lỗi khi tạo truyện cười: {str(e)}\n\nVui lòng kiểm tra cấu hình Azure OpenAI."

def create_gradio_interface():
    """
    Create and configure the Gradio interface for joke generation
    """
    demo = gr.Interface(
        fn=generate_joke,
        inputs=[
            gr.Textbox(
                label="Chủ đề truyện cười",
                placeholder="Nhập chủ đề cho truyện cười (ví dụ: động vật, học tập, công nghệ...)...",
                lines=2,
                max_lines=3
            ),
            gr.Dropdown(
                label="Ngôn ngữ",
                choices=["English", "Vietnamese", "Chinese"],
                value="Vietnamese"
            ),
            gr.Slider(
                label="Độ tuổi",
                value=18,
                minimum=5,
                maximum=99,
                step=1
            )
        ],
        outputs=gr.Textbox(
            label="Truyện cười được tạo",
            lines=8,
            max_lines=15
        ),
        title="🎭 Ứng dụng tạo truyện cười bằng AI",
        description="Tạo truyện cười vui nhộn bằng trí tuệ nhân tạo. Chọn chủ đề, ngôn ngữ và độ tuổi phù hợp để có những câu chuyện cười hay nhất!",
        theme=gr.themes.Soft(),
        examples=[
            ["động vật", "Vietnamese", 12],
            ["school life", "English", 16],
            ["technology", "English", 25],
            ["家庭生活", "Chinese", 30],
            ["du lịch", "Vietnamese", 20],
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
