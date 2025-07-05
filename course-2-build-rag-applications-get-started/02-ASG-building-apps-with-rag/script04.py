"""
Personal Financial Advisor Gradio Application using LLM
================================================================================

Module: script04.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates a personal financial advisor application using Azure OpenAI LLM
    and Gradio. It provides personalized financial advice through an easy-to-use
    web interface, generating recommendations based on personal information and financial status.
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
    max_tokens=2000
)

def generate_financial_advice(description, age, has_family, num_children, current_salary):
    """
    Generate personalized financial advice based on user's personal and financial information
    """
    if not description.strip():
        return "Vui lòng cung cấp mô tả về bản thân để nhận được tư vấn phù hợp."
    
    if not current_salary or current_salary <= 0:
        return "Vui lòng nhập mức lương hiện tại hợp lệ."
    
    try:
        # Format salary for display
        salary_formatted = f"{current_salary:,} VND"
        
        # Determine life stage and responsibilities
        family_status = "đã có gia đình" if has_family else "chưa có gia đình"
        children_info = f"có {num_children} con" if has_family and num_children > 0 else "chưa có con"
        
        # Create age-specific advice categories
        if age < 25:
            life_stage = "giai đoạn bắt đầu sự nghiệp"
            priorities = "tích lũy, học hỏi và đầu tư vào bản thân"
        elif age < 35:
            life_stage = "giai đoạn phát triển sự nghiệp"
            priorities = "tích lũy tài sản, mua nhà và đầu tư dài hạn"
        elif age < 50:
            life_stage = "giai đoạn ổn định và tích lũy"
            priorities = "tối ưu hóa đầu tư, chuẩn bị cho tương lai con cái"
        else:
            life_stage = "giai đoạn chuẩn bị nghỉ hưu"
            priorities = "bảo toàn tài sản và chuẩn bị hưu trí"
        
        prompt = f"""
        Bạn là một chuyên gia tư vấn tài chính cá nhân chuyên nghiệp tại Việt Nam. 
        Hãy đưa ra lời khuyên tài chính cá nhân chi tiết và thực tế dựa trên thông tin sau:

        THÔNG TIN CÁ NHÂN:
        - Mô tả: {description}
        - Tuổi: {age} tuổi ({life_stage})
        - Tình trạng gia đình: {family_status}
        - Số con: {children_info}
        - Mức lương hiện tại: {salary_formatted}

        YÊU CẦU TU VẤN:
        1. Phân tích tình hình tài chính hiện tại
        2. Đưa ra kế hoạch phân bổ thu nhập (50/30/20 rule hoặc điều chỉnh phù hợp)
        3. Gợi ý về tiết kiệm và đầu tư phù hợp với độ tuổi và hoàn cảnh
        4. Lời khuyên về bảo hiểm và quỹ dự phòng
        5. Kế hoạch tài chính dài hạn ({priorities})
        6. Những lưu ý đặc biệt dựa trên mô tả cá nhân

        NGUYÊN TẮC:
        - Đưa ra lời khuyên thực tế, phù hợp với điều kiện Việt Nam
        - Sử dụng số liệu cụ thể khi có thể
        - Ưu tiên tính an toàn và bền vững
        - Tránh các khuyến nghị đầu tư có rủi ro cao
        - Sử dụng tiếng Việt tự nhiên và dễ hiểu

        Hãy trả lời một cách chi tiết, có cấu trúc và thực tế.
        """
        
        # Send prompt to LLM
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        return f"Lỗi khi tạo tư vấn tài chính: {str(e)}\n\nVui lòng kiểm tra cấu hình Azure OpenAI."

def create_gradio_interface():
    """
    Create and configure the Gradio interface for financial advisory
    """
    with gr.Blocks(theme=gr.themes.Soft(), title="💰 Ứng dụng tư vấn tài chính cá nhân bằng AI") as demo:
        gr.Markdown("# 💰 Ứng dụng tư vấn tài chính cá nhân bằng AI")
        gr.Markdown("Nhận tư vấn tài chính cá nhân chuyên nghiệp từ trí tuệ nhân tạo. Cung cấp thông tin cá nhân để nhận được lời khuyên tài chính phù hợp với hoàn cảnh của bạn!")
        
        with gr.Row():
            with gr.Column(scale=1):
                description = gr.Textbox(
                    label="Mô tả về bạn",
                    placeholder="Hãy mô tả về bản thân (nghề nghiệp, mục tiêu tài chính, tình hình hiện tại...)...",
                    lines=4,
                    max_lines=6
                )
                
                age = gr.Slider(
                    label="Tuổi",
                    value=30,
                    minimum=18,
                    maximum=70,
                    step=1
                )
                
                has_family = gr.Checkbox(
                    label="Đã có gia đình",
                    value=False
                )
                
                num_children = gr.Slider(
                    label="Số con",
                    value=0,
                    minimum=0,
                    maximum=10,
                    step=1
                )
                
                current_salary = gr.Slider(
                    label="Mức lương hiện tại (VND)",
                    value=15000000,
                    minimum=5000000,
                    maximum=100000000,
                    step=1000000
                )
                
                submit_btn = gr.Button("Tư vấn tài chính", variant="primary")
                
            with gr.Column(scale=2):
                advice_output = gr.Textbox(
                    label="Tư vấn tài chính cá nhân",
                    lines=20,
                    max_lines=30
                )
        
        # Submit button click
        submit_btn.click(
            fn=generate_financial_advice,
            inputs=[description, age, has_family, num_children, current_salary],
            outputs=[advice_output]
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["Tôi là kỹ sư phần mềm, muốn mua nhà trong 3 năm tới và có kế hoạch kết hôn", 28, False, 0, 25000000],
                ["Tôi làm giáo viên, vừa có con đầu lòng, muốn tích lũy cho tương lai con", 32, True, 1, 12000000],
                ["Tôi là doanh nhân, muốn mở rộng kinh doanh và đầu tư bất động sản", 40, True, 2, 50000000],
                ["Tôi làm nhân viên văn phòng, muốn chuẩn bị nghỉ hưu sớm", 45, True, 1, 20000000],
                ["Tôi mới tốt nghiệp, bắt đầu đi làm và muốn học cách quản lý tài chính", 23, False, 0, 8000000],
            ],
            inputs=[description, age, has_family, num_children, current_salary],
            outputs=[advice_output]
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

