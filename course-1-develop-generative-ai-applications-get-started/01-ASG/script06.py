#!/usr/bin/env python3
"""
Prompt Engineering Demo: Designing prompts with clear instructions and rich context
Minh họa cách thiết kế prompts với hướng dẫn rõ ràng và ngữ cảnh phong phú
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()

print("=== PROMPT ENGINEERING: Clear Instructions & Rich Context ===\n")

# Setup Azure OpenAI connection
print("1. Kết nối với Azure OpenAI...")
try:
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0.3,
        max_tokens=400
    )
    print("✓ Kết nối thành công")
except Exception as e:
    print(f"✗ Lỗi kết nối: {e}")
    exit(1)

print("\n" + "="*80)

# DEMO 1: VAGUE vs CLEAR INSTRUCTIONS
print("DEMO 1: SO SÁNH PROMPT MƠHỒ vs RÕ RÀNG\n")

# BAD EXAMPLE: Vague prompt
print("❌ PROMPT MƠHỒ (Không nên):")
vague_prompt = "Viết về AI"
print(f"Prompt: '{vague_prompt}'")
print("Đang xử lý...")

try:
    response = llm.invoke(vague_prompt)
    print(f"Kết quả: {response.content}")
except Exception as e:
    print(f"Lỗi: {e}")

print("\n" + "-"*50)

# GOOD EXAMPLE: Clear instructions
print("✅ PROMPT RÕ RÀNG (Nên làm):")
clear_prompt = """
Viết một đoạn văn ngắn (khoảng 100 từ) về AI cho học sinh cấp 3.
Yêu cầu:
- Giải thích AI là gì một cách đơn giản
- Đưa ra 2 ví dụ cụ thể về AI trong đời sống
- Sử dụng ngôn ngữ dễ hiểu, không quá kỹ thuật
- Kết thúc bằng một câu tích cực về tương lai AI
"""
print(f"Prompt: {clear_prompt}")
print("Đang xử lý...")

try:
    response = llm.invoke(clear_prompt)
    print(f"Kết quả: {response.content}")
except Exception as e:
    print(f"Lỗi: {e}")

print("\n" + "="*80)

# DEMO 2: NO CONTEXT vs RICH CONTEXT
print("DEMO 2: SO SÁNH KHÔNG NGỮCẢNH vs NGỮCẢNH PHONG PHÚ\n")

# BAD EXAMPLE: No context
print("❌ KHÔNG NGỮCẢNH (Không nên):")
no_context_prompt = "Tôi nên làm gì?"
print(f"Prompt: '{no_context_prompt}'")
print("Đang xử lý...")

try:
    response = llm.invoke(no_context_prompt)
    print(f"Kết quả: {response.content}")
except Exception as e:
    print(f"Lỗi: {e}")

print("\n" + "-"*50)

# GOOD EXAMPLE: Rich context
print("✅ NGỮCẢNH PHONG PHÚ (Nên làm):")
rich_context_prompt = """
NGỮCẢNH: Tôi là một sinh viên năm 3 ngành Công nghệ thông tin. Tôi đang học về AI và muốn tìm một dự án thực tế để thực hiện trong học kỳ này.

THÔNG TIN BỔ SUNG:
- Thời gian: 3 tháng
- Kỹ năng hiện tại: Python cơ bản, đã học machine learning lý thuyết
- Mục tiêu: Tạo ra sản phẩm có thể demo được
- Sở thích: Game, âm nhạc, thể thao

YÊU CẦU: Hãy đề xuất 3 ý tưởng dự án AI phù hợp với tình hình của tôi. Mỗi ý tưởng cần bao gồm:
1. Tên dự án
2. Mô tả ngắn gọn
3. Công nghệ sử dụng
4. Mức độ khó (1-10)
5. Kết quả mong đợi
"""
print(f"Prompt: {rich_context_prompt}")
print("Đang xử lý...")

try:
    response = llm.invoke(rich_context_prompt)
    print(f"Kết quả: {response.content}")
except Exception as e:
    print(f"Lỗi: {e}")

print("\n" + "="*80)

# DEMO 3: ROLE-BASED PROMPTING WITH CONTEXT
print("DEMO 3: PROMPT THEO VAI TRÒ VỚI NGỮCẢNH\n")

role_prompt = """
VAI TRÒ: Bạn là một chuyên gia tư vấn khách hàng của một ngân hàng với 10 năm kinh nghiệm.

NGỮCẢNH: Khách hàng là một cặp vợ chồng trẻ (28 tuổi), mới cưới, đang có thu nhập ổn định 50 triệu/tháng, muốn mua nhà đầu tiên.

THÔNG TIN KHÁCH HÀNG:
- Tiết kiệm hiện tại: 800 triệu
- Mức nhà mong muốn: 3-4 tỷ  
- Vị trí: Gần trung tâm TP.HCM
- Mục tiêu: Ổn định lâu dài, không muốn áp lực tài chính quá lớn

NHIỆM VỤ: Hãy đưa ra lời khuyên chi tiết về:
1. Chiến lược tài chính (tỷ lệ vay/vốn tự có)
2. Loại hình vay phù hợp
3. Những lưu ý quan trọng khi mua nhà
4. Timeline thực hiện

YÊU CẦU: Trả lời theo phong cách chuyên nghiệp nhưng thân thiện, dễ hiểu.
"""
print(f"Prompt: {role_prompt}")
print("Đang xử lý...")

try:
    response = llm.invoke(role_prompt)
    print(f"Kết quả: {response.content}")
except Exception as e:
    print(f"Lỗi: {e}")

print("\n" + "="*80)

# DEMO 4: STRUCTURED OUTPUT WITH CONSTRAINTS
print("DEMO 4: YÊU CẦU ĐỊNH DẠNG ĐẦU RA CỤ THỂ\n")

structured_prompt = """
NHIỆM VỤ: Phân tích SWOT cho công ty khởi nghiệp về ứng dụng giao đồ ăn.

NGỮCẢNH:
- Công ty: FoodExpress
- Thị trường: Việt Nam
- Giai đoạn: Startup mới thành lập
- Đối thủ chính: Grab Food, Shopee Food, Baemin

YÊU CẦU ĐỊNH DẠNG:
Strengths (Điểm mạnh):
- [Điểm 1]: [Giải thích ngắn]
- [Điểm 2]: [Giải thích ngắn]
- [Điểm 3]: [Giải thích ngắn]

Weaknesses (Điểm yếu):
- [Điểm 1]: [Giải thích ngắn]
- [Điểm 2]: [Giải thích ngắn]
- [Điểm 3]: [Giải thích ngắn]

Opportunities (Cơ hội):
- [Cơ hội 1]: [Giải thích ngắn]
- [Cơ hội 2]: [Giải thích ngắn]
- [Cơ hội 3]: [Giải thích ngắn]

Threats (Thách thức):
- [Thách thức 1]: [Giải thích ngắn]
- [Thách thức 2]: [Giải thích ngắn]
- [Thách thức 3]: [Giải thích ngắn]

KHUYẾN NGHỊ CHIẾN LƯỢC:
[2-3 câu tổng kết và đề xuất hướng phát triển]

LƯU Ý: Mỗi mục chỉ nên 1-2 câu, tập trung vào thông tin quan trọng nhất.
"""
print(f"Prompt: {structured_prompt}")
print("Đang xử lý...")

try:
    response = llm.invoke(structured_prompt)
    print(f"Kết quả: {response.content}")
except Exception as e:
    print(f"Lỗi: {e}")

print("\n" + "="*80)
print("✅ DEMO HOÀN THÀNH!")
print("\nCÁC NGUYÊN TẮC PROMPT ENGINEERING ĐÃ MINH HỌA:")
print("1. 🎯 Hướng dẫn rõ ràng thay vì mơ hồ")
print("2. 📖 Cung cấp ngữ cảnh phong phú")
print("3. 🎭 Sử dụng vai trò cụ thể")
print("4. 📋 Yêu cầu định dạng đầu ra")
print("5. ⚡ Ràng buộc và giới hạn rõ ràng")
print("\n💡 KẾT LUẬN: Prompt tốt = Hướng dẫn rõ ràng + Ngữ cảnh phong phú = Kết quả chất lượng cao!")
