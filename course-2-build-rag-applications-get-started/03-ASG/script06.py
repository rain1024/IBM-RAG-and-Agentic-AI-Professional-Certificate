"""
Gradio CheckboxGroup Demo Application
================================================================================

Module: script06.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates a Gradio application showcasing CheckboxGroup
    functionality. It provides examples of how to use CheckboxGroup in
    different scenarios including food preferences, skills selection,
    and language preferences.
"""

import gradio as gr

def process_food_preferences(selected_foods):
    """
    Process selected food preferences
    """
    if not selected_foods:
        return "Bạn chưa chọn món ăn nào!"
    
    result = f"✅ Bạn đã chọn {len(selected_foods)} món ăn:\n"
    for food in selected_foods:
        result += f"• {food}\n"
    
    # Add some recommendations based on selections
    recommendations = []
    if "Pizza" in selected_foods:
        recommendations.append("Thử pizza Margherita với phô mai tươi!")
    if "Sushi" in selected_foods:
        recommendations.append("Gợi ý: Sashimi cá hồi rất tươi ngon!")
    if "Phở" in selected_foods:
        recommendations.append("Phở bò tái chín với rau thơm là tuyệt vời!")
    
    if recommendations:
        result += "\n💡 Gợi ý cho bạn:\n"
        for rec in recommendations:
            result += f"• {rec}\n"
    
    return result

def process_skills(selected_skills):
    """
    Process selected programming skills
    """
    if not selected_skills:
        return "Bạn chưa chọn kỹ năng nào!"
    
    result = f"🔧 Kỹ năng của bạn ({len(selected_skills)} kỹ năng):\n"
    
    # Categorize skills
    frontend_skills = ["HTML/CSS", "JavaScript", "React", "Vue.js"]
    backend_skills = ["Python", "Java", "Node.js", "PHP"]
    database_skills = ["SQL", "MongoDB", "PostgreSQL"]
    
    frontend_selected = [skill for skill in selected_skills if skill in frontend_skills]
    backend_selected = [skill for skill in selected_skills if skill in backend_skills]
    database_selected = [skill for skill in selected_skills if skill in database_skills]
    
    if frontend_selected:
        result += f"\n🎨 Frontend: {', '.join(frontend_selected)}"
    if backend_selected:
        result += f"\n⚙️ Backend: {', '.join(backend_selected)}"
    if database_selected:
        result += f"\n🗄️ Database: {', '.join(database_selected)}"
    
    # Career suggestions
    if len(selected_skills) >= 5:
        result += "\n\n🚀 Bạn có kỹ năng toàn diện! Có thể làm Full-stack Developer."
    elif frontend_selected and not backend_selected:
        result += "\n\n💡 Gợi ý: Học thêm backend để trở thành Full-stack Developer!"
    elif backend_selected and not frontend_selected:
        result += "\n\n💡 Gợi ý: Học thêm frontend để mở rộng kỹ năng!"
    
    return result

def process_languages(selected_languages):
    """
    Process selected languages
    """
    if not selected_languages:
        return "Bạn chưa chọn ngôn ngữ nào!"
    
    result = f"🌍 Ngôn ngữ bạn biết ({len(selected_languages)} ngôn ngữ):\n"
    
    # Language info
    language_info = {
        "Tiếng Việt": "🇻🇳 Ngôn ngữ mẹ đẻ",
        "English": "🇺🇸 Ngôn ngữ quốc tế",
        "中文": "🇨🇳 Ngôn ngữ có nhiều người nói nhất",
        "日本語": "🇯🇵 Ngôn ngữ công nghệ",
        "한국어": "🇰🇷 Ngôn ngữ K-culture",
        "Français": "🇫🇷 Ngôn ngữ lãng mạn",
        "Español": "🇪🇸 Ngôn ngữ Latin",
        "Deutsch": "🇩🇪 Ngôn ngữ kỹ thuật"
    }
    
    for lang in selected_languages:
        info = language_info.get(lang, "🌐 Ngôn ngữ quốc tế")
        result += f"• {lang} - {info}\n"
    
    # Benefits
    if len(selected_languages) >= 4:
        result += "\n🎉 Tuyệt vời! Bạn là một polyglot thực thụ!"
    elif len(selected_languages) >= 2:
        result += "\n✨ Bạn có lợi thế lớn trong giao tiếp quốc tế!"
    
    return result

def create_gradio_interface():
    """
    Create and configure the Gradio interface for CheckboxGroup demo
    """
    with gr.Blocks(theme=gr.themes.Soft(), title="CheckboxGroup Demo") as demo:
        gr.Markdown("""
        # 📋 Gradio CheckboxGroup Demo
        
        Ứng dụng demo các chức năng của CheckboxGroup trong Gradio.
        Chọn các mục bạn thích và xem kết quả được xử lý!
        """)
        
        with gr.Tab("🍕 Sở thích ẩm thực"):
            gr.Markdown("### Chọn những món ăn bạn yêu thích:")
            
            food_checkbox = gr.CheckboxGroup(
                label="Món ăn yêu thích",
                choices=[
                    "Pizza", "Sushi", "Phở", "Bánh mì", "Hamburger",
                    "Pasta", "Ramen", "Bánh cuốn", "Tacos", "Dim Sum"
                ],
                value=["Pizza", "Phở"],
                interactive=True
            )
            
            food_button = gr.Button("Xử lý sở thích", variant="primary")
            food_output = gr.Textbox(
                label="Kết quả",
                lines=10,
                interactive=False
            )
            
            food_button.click(
                fn=process_food_preferences,
                inputs=[food_checkbox],
                outputs=[food_output]
            )
        
        with gr.Tab("💻 Kỹ năng lập trình"):
            gr.Markdown("### Chọn những kỹ năng lập trình bạn có:")
            
            skills_checkbox = gr.CheckboxGroup(
                label="Kỹ năng lập trình",
                choices=[
                    "Python", "JavaScript", "Java", "HTML/CSS", 
                    "React", "Vue.js", "Node.js", "PHP", 
                    "SQL", "MongoDB", "PostgreSQL", "Docker"
                ],
                value=["Python", "JavaScript"],
                interactive=True
            )
            
            skills_button = gr.Button("Phân tích kỹ năng", variant="primary")
            skills_output = gr.Textbox(
                label="Phân tích kỹ năng",
                lines=10,
                interactive=False
            )
            
            skills_button.click(
                fn=process_skills,
                inputs=[skills_checkbox],
                outputs=[skills_output]
            )
        
        with gr.Tab("🌍 Ngôn ngữ"):
            gr.Markdown("### Chọn những ngôn ngữ bạn biết:")
            
            lang_checkbox = gr.CheckboxGroup(
                label="Ngôn ngữ",
                choices=[
                    "Tiếng Việt", "English", "中文", "日本語", 
                    "한국어", "Français", "Español", "Deutsch"
                ],
                value=["Tiếng Việt", "English"],
                interactive=True
            )
            
            lang_button = gr.Button("Phân tích ngôn ngữ", variant="primary")
            lang_output = gr.Textbox(
                label="Phân tích ngôn ngữ",
                lines=10,
                interactive=False
            )
            
            lang_button.click(
                fn=process_languages,
                inputs=[lang_checkbox],
                outputs=[lang_output]
            )
        
        with gr.Tab("ℹ️ Hướng dẫn"):
            gr.Markdown("""
            ## Cách sử dụng CheckboxGroup
            
            ### 1. Tạo CheckboxGroup cơ bản:
            ```python
            gr.CheckboxGroup(
                label="Nhãn",
                choices=["Lựa chọn 1", "Lựa chọn 2", "Lựa chọn 3"],
                value=["Lựa chọn 1"],  # Giá trị mặc định
                interactive=True
            )
            ```
            
            ### 2. Các tham số quan trọng:
            - **label**: Nhãn hiển thị
            - **choices**: Danh sách các lựa chọn
            - **value**: Giá trị mặc định được chọn
            - **interactive**: Cho phép tương tác
            - **visible**: Hiển thị/ẩn component
            
            ### 3. Xử lý dữ liệu:
            - Dữ liệu trả về là một **list** chứa các item được chọn
            - Có thể xử lý theo từng loại hoặc tổng hợp
            
            ### 4. Ứng dụng thực tế:
            - Form khảo sát
            - Chọn sở thích
            - Cấu hình hệ thống
            - Filter dữ liệu
            """)
    
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

