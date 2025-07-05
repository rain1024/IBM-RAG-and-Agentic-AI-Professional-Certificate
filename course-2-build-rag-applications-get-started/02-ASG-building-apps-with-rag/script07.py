"""
Gradio Dropdown Demo Application
================================================================================

Module: script07.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    This module demonstrates a Gradio application showcasing Dropdown
    functionality. It provides examples of how to use Dropdown in
    different scenarios including food selection, skills showcase,
    and language preference.
"""

import gradio as gr

def process_food_selection(selected_food):
    """
    Process selected food preference
    """
    if not selected_food:
        return "Bạn chưa chọn món ăn nào!"
    
    # Food information database
    food_info = {
        "Pizza": {
            "origin": "🇮🇹 Ý",
            "description": "Bánh pizza giòn với phô mai và sốt cà chua",
            "price": "150,000 - 300,000 VND",
            "tips": "Nên ăn khi còn nóng, kết hợp với nước ngọt"
        },
        "Sushi": {
            "origin": "🇯🇵 Nhật Bản",
            "description": "Cơm trộn giấm với hải sản tươi",
            "price": "200,000 - 500,000 VND",
            "tips": "Ăn với wasabi và nước tương, uống trà xanh"
        },
        "Phở": {
            "origin": "🇻🇳 Việt Nam",
            "description": "Món nước truyền thống với bánh phở và thịt bò",
            "price": "50,000 - 100,000 VND",
            "tips": "Ăn kèm rau thơm, chanh và tương ớt"
        },
        "Bánh mì": {
            "origin": "🇻🇳 Việt Nam",
            "description": "Bánh mì giòn với nhân thịt và rau củ",
            "price": "20,000 - 50,000 VND",
            "tips": "Ăn khi bánh còn giòn, có thể kết hợp với cà phê"
        },
        "Hamburger": {
            "origin": "🇺🇸 Mỹ",
            "description": "Bánh mì kẹp thịt với rau và sốt",
            "price": "80,000 - 200,000 VND",
            "tips": "Ăn kèm khoai tây chiên và nước ngọt"
        },
        "Pasta": {
            "origin": "🇮🇹 Ý",
            "description": "Mì Ý với nhiều loại sốt khác nhau",
            "price": "120,000 - 250,000 VND",
            "tips": "Ăn kèm phô mai Parmesan và rượu vang"
        },
        "Ramen": {
            "origin": "🇯🇵 Nhật Bản",
            "description": "Mì ramen trong nước dùng đậm đà",
            "price": "100,000 - 200,000 VND",
            "tips": "Ăn nóng, có thể thêm trứng và rau"
        },
        "Bánh cuốn": {
            "origin": "🇻🇳 Việt Nam",
            "description": "Bánh tráng mỏng cuốn nhân thịt",
            "price": "30,000 - 60,000 VND",
            "tips": "Ăn kèm chả lụa và nước mắm pha"
        },
        "Tacos": {
            "origin": "🇲🇽 Mexico",
            "description": "Bánh tortilla cuốn thịt và rau",
            "price": "60,000 - 120,000 VND",
            "tips": "Ăn kèm sốt salsa và kem chua"
        },
        "Dim Sum": {
            "origin": "🇨🇳 Trung Quốc",
            "description": "Các món dim sum nhỏ đa dạng",
            "price": "80,000 - 150,000 VND",
            "tips": "Ăn kèm trà Oolong, thích hợp ăn sáng"
        }
    }
    
    info = food_info.get(selected_food, {})
    
    result = f"🍽️ Bạn đã chọn: **{selected_food}**\n\n"
    
    if info:
        result += f"📍 **Xuất xứ:** {info['origin']}\n"
        result += f"📝 **Mô tả:** {info['description']}\n"
        result += f"💰 **Giá tham khảo:** {info['price']}\n"
        result += f"💡 **Mẹo:** {info['tips']}\n"
    
    return result

def process_skill_selection(selected_skill):
    """
    Process selected programming skill
    """
    if not selected_skill:
        return "Bạn chưa chọn kỹ năng nào!"
    
    # Skill information database
    skill_info = {
        "Python": {
            "category": "🐍 Backend/Data Science",
            "difficulty": "⭐⭐⭐ Trung bình",
            "salary": "20-40 triệu/tháng",
            "description": "Ngôn ngữ lập trình đa năng, mạnh về AI/ML",
            "learning_path": "Cơ bản → Django/Flask → Data Science → AI/ML"
        },
        "JavaScript": {
            "category": "🌐 Frontend/Backend",
            "difficulty": "⭐⭐⭐ Trung bình",
            "salary": "18-35 triệu/tháng",
            "description": "Ngôn ngữ web phổ biến nhất",
            "learning_path": "ES6 → React/Vue → Node.js → Full-stack"
        },
        "Java": {
            "category": "☕ Backend/Enterprise",
            "difficulty": "⭐⭐⭐⭐ Khó",
            "salary": "22-45 triệu/tháng",
            "description": "Ngôn ngữ doanh nghiệp, mạnh về backend",
            "learning_path": "OOP → Spring → Microservices → Cloud"
        },
        "HTML/CSS": {
            "category": "🎨 Frontend",
            "difficulty": "⭐⭐ Dễ",
            "salary": "12-25 triệu/tháng",
            "description": "Nền tảng phát triển web",
            "learning_path": "HTML5 → CSS3 → Responsive → Framework"
        },
        "React": {
            "category": "⚛️ Frontend Framework",
            "difficulty": "⭐⭐⭐ Trung bình",
            "salary": "20-40 triệu/tháng",
            "description": "Framework phổ biến nhất để xây dựng UI",
            "learning_path": "Components → Hooks → Redux → Next.js"
        },
        "Vue.js": {
            "category": "💚 Frontend Framework",
            "difficulty": "⭐⭐ Dễ",
            "salary": "18-35 triệu/tháng",
            "description": "Framework dễ học, linh hoạt",
            "learning_path": "Template → Components → Vuex → Nuxt.js"
        },
        "Node.js": {
            "category": "🚀 Backend Runtime",
            "difficulty": "⭐⭐⭐ Trung bình",
            "salary": "20-38 triệu/tháng",
            "description": "Chạy JavaScript trên server",
            "learning_path": "Express → Database → API → Microservices"
        },
        "PHP": {
            "category": "🐘 Backend",
            "difficulty": "⭐⭐ Dễ",
            "salary": "15-30 triệu/tháng",
            "description": "Ngôn ngữ web truyền thống",
            "learning_path": "Cơ bản → Laravel → Database → CMS"
        },
        "SQL": {
            "category": "🗄️ Database",
            "difficulty": "⭐⭐⭐ Trung bình",
            "salary": "18-35 triệu/tháng",
            "description": "Ngôn ngữ truy vấn cơ sở dữ liệu",
            "learning_path": "SELECT → JOIN → Stored Procedure → Optimization"
        },
        "MongoDB": {
            "category": "🍃 NoSQL Database",
            "difficulty": "⭐⭐⭐ Trung bình",
            "salary": "20-40 triệu/tháng",
            "description": "Cơ sở dữ liệu NoSQL phổ biến",
            "learning_path": "CRUD → Aggregation → Indexing → Sharding"
        },
        "PostgreSQL": {
            "category": "🐘 SQL Database",
            "difficulty": "⭐⭐⭐⭐ Khó",
            "salary": "22-42 triệu/tháng",
            "description": "Cơ sở dữ liệu quan hệ mạnh mẽ",
            "learning_path": "SQL → Advanced Features → Performance → Admin"
        },
        "Docker": {
            "category": "🐳 DevOps",
            "difficulty": "⭐⭐⭐ Trung bình",
            "salary": "25-50 triệu/tháng",
            "description": "Containerization platform",
            "learning_path": "Images → Containers → Compose → Kubernetes"
        }
    }
    
    info = skill_info.get(selected_skill, {})
    
    result = f"🔧 Kỹ năng bạn chọn: **{selected_skill}**\n\n"
    
    if info:
        result += f"📂 **Danh mục:** {info['category']}\n"
        result += f"📊 **Độ khó:** {info['difficulty']}\n"
        result += f"💰 **Mức lương:** {info['salary']}\n"
        result += f"📝 **Mô tả:** {info['description']}\n"
        result += f"🎯 **Lộ trình học:** {info['learning_path']}\n"
    
    return result

def process_language_selection(selected_language):
    """
    Process selected language
    """
    if not selected_language:
        return "Bạn chưa chọn ngôn ngữ nào!"
    
    # Language information database
    language_info = {
        "Tiếng Việt": {
            "flag": "🇻🇳",
            "speakers": "95 triệu người",
            "difficulty": "⭐⭐⭐ Trung bình (cho người nước ngoài)",
            "benefits": "Ngôn ngữ mẹ đẻ, thuận lợi trong công việc tại VN",
            "career": "Tất cả các ngành nghề tại Việt Nam"
        },
        "English": {
            "flag": "🇺🇸",
            "speakers": "1.5 tỷ người",
            "difficulty": "⭐⭐⭐ Trung bình",
            "benefits": "Ngôn ngữ quốc tế, cơ hội việc làm toàn cầu",
            "career": "IT, Kinh doanh quốc tế, Du lịch, Giáo dục"
        },
        "中文": {
            "flag": "🇨🇳",
            "speakers": "1.4 tỷ người",
            "difficulty": "⭐⭐⭐⭐⭐ Rất khó",
            "benefits": "Thị trường lớn nhất thế giới, cơ hội kinh doanh",
            "career": "Thương mại, Sản xuất, Logistics, Du lịch"
        },
        "日本語": {
            "flag": "🇯🇵",
            "speakers": "125 triệu người",
            "difficulty": "⭐⭐⭐⭐⭐ Rất khó",
            "benefits": "Công nghệ cao, văn hóa anime/manga",
            "career": "IT, Kỹ thuật, Dịch thuật, Du lịch"
        },
        "한국어": {
            "flag": "🇰🇷",
            "speakers": "77 triệu người",
            "difficulty": "⭐⭐⭐⭐ Khó",
            "benefits": "Hallyu Wave, công nghệ, mỹ phẩm",
            "career": "Giải trí, Công nghệ, Mỹ phẩm, Du lịch"
        },
        "Français": {
            "flag": "🇫🇷",
            "speakers": "280 triệu người",
            "difficulty": "⭐⭐⭐⭐ Khó",
            "benefits": "Văn hóa, thời trang, ẩm thực",
            "career": "Thời trang, Ẩm thực, Du lịch, Ngoại giao"
        },
        "Español": {
            "flag": "🇪🇸",
            "speakers": "500 triệu người",
            "difficulty": "⭐⭐⭐ Trung bình",
            "benefits": "Ngôn ngữ phổ biến thứ 2 thế giới",
            "career": "Du lịch, Thương mại, Giáo dục, Dịch thuật"
        },
        "Deutsch": {
            "flag": "🇩🇪",
            "speakers": "100 triệu người",
            "difficulty": "⭐⭐⭐⭐ Khó",
            "benefits": "Kinh tế mạnh, kỹ thuật, khoa học",
            "career": "Kỹ thuật, Ô tô, Khoa học, Giáo dục"
        }
    }
    
    info = language_info.get(selected_language, {})
    
    result = f"🌍 Ngôn ngữ bạn chọn: **{selected_language}**\n\n"
    
    if info:
        result += f"🏳️ **Quốc gia:** {info['flag']}\n"
        result += f"👥 **Người sử dụng:** {info['speakers']}\n"
        result += f"📈 **Độ khó:** {info['difficulty']}\n"
        result += f"✨ **Lợi ích:** {info['benefits']}\n"
        result += f"💼 **Cơ hội nghề nghiệp:** {info['career']}\n"
    
    return result

def create_gradio_interface():
    """
    Create and configure the Gradio interface for Dropdown demo
    """
    with gr.Blocks(theme=gr.themes.Soft(), title="Dropdown Demo") as demo:
        gr.Markdown("""
        # 📋 Gradio Dropdown Demo
        
        Ứng dụng demo các chức năng của Dropdown trong Gradio.
        Chọn một mục bạn quan tâm và xem thông tin chi tiết!
        """)
        
        with gr.Tab("🍕 Chọn món ăn"):
            gr.Markdown("### Chọn một món ăn bạn muốn tìm hiểu:")
            
            food_dropdown = gr.Dropdown(
                label="Món ăn",
                choices=[
                    "Pizza", "Sushi", "Phở", "Bánh mì", "Hamburger",
                    "Pasta", "Ramen", "Bánh cuốn", "Tacos", "Dim Sum"
                ],
                value="Pizza",
                interactive=True
            )
            
            food_button = gr.Button("Xem thông tin món ăn", variant="primary")
            food_output = gr.Textbox(
                label="Thông tin món ăn",
                lines=8,
                interactive=False
            )
            
            food_button.click(
                fn=process_food_selection,
                inputs=[food_dropdown],
                outputs=[food_output]
            )
        
        with gr.Tab("💻 Kỹ năng lập trình"):
            gr.Markdown("### Chọn một kỹ năng lập trình để tìm hiểu:")
            
            skills_dropdown = gr.Dropdown(
                label="Kỹ năng lập trình",
                choices=[
                    "Python", "JavaScript", "Java", "HTML/CSS", 
                    "React", "Vue.js", "Node.js", "PHP", 
                    "SQL", "MongoDB", "PostgreSQL", "Docker"
                ],
                value="Python",
                interactive=True
            )
            
            skills_button = gr.Button("Xem thông tin kỹ năng", variant="primary")
            skills_output = gr.Textbox(
                label="Thông tin kỹ năng",
                lines=8,
                interactive=False
            )
            
            skills_button.click(
                fn=process_skill_selection,
                inputs=[skills_dropdown],
                outputs=[skills_output]
            )
        
        with gr.Tab("🌍 Ngôn ngữ"):
            gr.Markdown("### Chọn một ngôn ngữ để tìm hiểu:")
            
            lang_dropdown = gr.Dropdown(
                label="Ngôn ngữ",
                choices=[
                    "Tiếng Việt", "English", "中文", "日本語", 
                    "한국어", "Français", "Español", "Deutsch"
                ],
                value="Tiếng Việt",
                interactive=True
            )
            
            lang_button = gr.Button("Xem thông tin ngôn ngữ", variant="primary")
            lang_output = gr.Textbox(
                label="Thông tin ngôn ngữ",
                lines=8,
                interactive=False
            )
            
            lang_button.click(
                fn=process_language_selection,
                inputs=[lang_dropdown],
                outputs=[lang_output]
            )
        
        with gr.Tab("ℹ️ Hướng dẫn"):
            gr.Markdown("""
            ## Cách sử dụng Dropdown
            
            ### 1. Tạo Dropdown cơ bản:
            ```python
            gr.Dropdown(
                label="Nhãn",
                choices=["Lựa chọn 1", "Lựa chọn 2", "Lựa chọn 3"],
                value="Lựa chọn 1",  # Giá trị mặc định
                interactive=True
            )
            ```
            
            ### 2. Các tham số quan trọng:
            - **label**: Nhãn hiển thị
            - **choices**: Danh sách các lựa chọn
            - **value**: Giá trị mặc định được chọn
            - **interactive**: Cho phép tương tác
            - **multiselect**: Cho phép chọn nhiều (mặc định False)
            - **allow_custom_value**: Cho phép nhập giá trị tùy chỉnh
            
            ### 3. Xử lý dữ liệu:
            - Dữ liệu trả về là **string** (1 giá trị được chọn)
            - Nếu multiselect=True thì trả về **list**
            
            ### 4. Ứng dụng thực tế:
            - Chọn danh mục
            - Menu điều hướng
            - Bộ lọc dữ liệu
            - Cài đặt hệ thống
            - Form đăng ký
            
            ### 5. So sánh với CheckboxGroup:
            - **Dropdown**: Chọn 1 hoặc ít item, gọn gàng
            - **CheckboxGroup**: Chọn nhiều item, hiển thị rõ ràng
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

