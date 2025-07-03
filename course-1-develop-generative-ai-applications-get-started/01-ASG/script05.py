#!/usr/bin/env python3
"""
In-Context Learning Demo với Azure OpenAI
Minh họa khái niệm in-context learning: khả năng học từ ví dụ trong prompt mà không cần training thêm
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Setup Azure OpenAI giống như script04
print("=== In-Context Learning Demo với Azure OpenAI ===\n")

print("1. Khởi tạo Azure OpenAI...")
try:
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0.3,  # Thấp hơn để có kết quả ổn định hơn
        max_tokens=300
    )
    print("✓ Azure OpenAI đã được khởi tạo thành công")
except Exception as e:
    print(f"✗ Lỗi khởi tạo Azure OpenAI: {e}")
    print("Vui lòng kiểm tra các biến môi trường")
    exit(1)

print("\n" + "="*80)

# Demo 1: Zero-shot vs Few-shot Learning
print("2. SO SÁNH ZERO-SHOT vs FEW-SHOT LEARNING\n")

# Zero-shot (không có ví dụ)
print("A. ZERO-SHOT (không có ví dụ):")
zero_shot_prompt = """
Phân loại cảm xúc của câu này thành: tích cực, tiêu cực, hoặc trung tính.

Câu: "Sản phẩm này thật tuyệt vời!"
Cảm xúc:"""

print("Prompt Zero-shot:")
print(zero_shot_prompt)

try:
    zero_shot_response = llm.invoke(zero_shot_prompt)
    print(f"Kết quả: {zero_shot_response.content}")
except Exception as e:
    print(f"Lỗi: {e}")

print("\n" + "-"*50)

# Few-shot (có ví dụ)
print("B. FEW-SHOT (có ví dụ):")
few_shot_prompt = """
Phân loại cảm xúc của câu thành: tích cực, tiêu cực, hoặc trung tính.

Ví dụ:
Câu: "Tôi rất thích món ăn này!"
Cảm xúc: tích cực

Câu: "Dịch vụ ở đây thật tệ."
Cảm xúc: tiêu cực  

Câu: "Thời tiết hôm nay bình thường."
Cảm xúc: trung tính

Bây giờ phân loại câu này:
Câu: "Sản phẩm này thật tuyệt vời!"
Cảm xúc:"""

print("Prompt Few-shot:")
print(few_shot_prompt)

try:
    few_shot_response = llm.invoke(few_shot_prompt)
    print(f"Kết quả: {few_shot_response.content}")
except Exception as e:
    print(f"Lỗi: {e}")

print("\n" + "="*80)

# Demo 2: Chain-of-Thought Reasoning
print("3. CHAIN-OF-THOUGHT REASONING\n")

print("A. Không có chain-of-thought:")
simple_math = """
Giải bài toán sau:
Một cửa hàng bán 15 chiếc áo vào buổi sáng, 23 chiếc vào buổi chiều, và 8 chiếc vào buổi tối. 
Hỏi cửa hàng đã bán tổng cộng bao nhiêu chiếc áo?

Đáp án:"""

print("Prompt đơn giản:")
print(simple_math)

try:
    simple_response = llm.invoke(simple_math)
    print(f"Kết quả: {simple_response.content}")
except Exception as e:
    print(f"Lỗi: {e}")

print("\n" + "-"*50)

print("B. Có chain-of-thought:")
cot_math = """
Giải bài toán từng bước:

Ví dụ:
Bài toán: Một cửa hàng bán 12 quyển sách vào buổi sáng và 18 quyển vào buổi chiều. Hỏi đã bán tổng cộng bao nhiêu quyển?
Lời giải: 
- Buổi sáng: 12 quyển
- Buổi chiều: 18 quyển  
- Tổng cộng: 12 + 18 = 30 quyển

Bây giờ giải bài toán này:
Bài toán: Một cửa hàng bán 15 chiếc áo vào buổi sáng, 23 chiếc vào buổi chiều, và 8 chiếc vào buổi tối. 
Hỏi cửa hàng đã bán tổng cộng bao nhiêu chiếc áo?

Lời giải:"""

print("Prompt với chain-of-thought:")
print(cot_math)

try:
    cot_response = llm.invoke(cot_math)
    print(f"Kết quả: {cot_response.content}")
except Exception as e:
    print(f"Lỗi: {e}")

print("\n" + "="*80)

# Demo 3: Task Adaptation - Dịch thuật
print("4. TASK ADAPTATION - DỊCH THUẬT\n")

translation_prompt = """
Dịch các câu sau từ tiếng Anh sang tiếng Việt:

Ví dụ:
English: "Hello, how are you?"
Vietnamese: "Xin chào, bạn có khỏe không?"

English: "Thank you very much."
Vietnamese: "Cảm ơn bạn rất nhiều."

English: "What time is it?"
Vietnamese: "Bây giờ là mấy giờ?"

Bây giờ dịch câu này:
English: "I love learning artificial intelligence."
Vietnamese:"""

print("Prompt dịch thuật:")
print(translation_prompt)

try:
    translation_response = llm.invoke(translation_prompt)
    print(f"Kết quả: {translation_response.content}")
except Exception as e:
    print(f"Lỗi: {e}")

print("\n" + "="*80)

# Demo 4: Code Generation Pattern
print("5. CODE GENERATION PATTERN\n")

code_prompt = """
Tạo code Python dựa trên mô tả:

Ví dụ:
Mô tả: "Tạo function tính tổng 2 số"
Code:
def tinh_tong(a, b):
    return a + b

Mô tả: "Tạo function kiểm tra số chẵn"
Code:
def la_so_chan(n):
    return n % 2 == 0

Bây giờ tạo code cho:
Mô tả: "Tạo function tìm số lớn nhất trong danh sách"
Code:"""

print("Prompt code generation:")
print(code_prompt)

try:
    code_response = llm.invoke(code_prompt)
    print(f"Kết quả: {code_response.content}")
except Exception as e:
    print(f"Lỗi: {e}")

print("\n" + "="*80)

# Demo 5: Multi-task với một prompt
print("6. MULTI-TASK LEARNING\n")

multi_task_prompt = """
Thực hiện nhiều tác vụ dựa trên ví dụ:

Ví dụ:
Input: "Tôi rất thích món phở này! Nó thật ngon."
Tasks:
1. Phân loại cảm xúc: tích cực
2. Đếm từ: 8 từ
3. Chủ đề: đồ ăn

Input: "Thời tiết hôm nay có mưa."
Tasks:
1. Phân loại cảm xúc: trung tính
2. Đếm từ: 5 từ
3. Chủ đề: thời tiết

Bây giờ phân tích:
Input: "Bộ phim này thật tệ, tôi không thích."
Tasks:
1. Phân loại cảm xúc:
2. Đếm từ:
3. Chủ đề:"""

print("Prompt multi-task:")
print(multi_task_prompt)

try:
    multi_response = llm.invoke(multi_task_prompt)
    print(f"Kết quả: {multi_response.content}")
except Exception as e:
    print(f"Lỗi: {e}")

print("\n" + "="*80)

# Tổng kết
print("7. TỔNG KẾT VỀ IN-CONTEXT LEARNING\n")

summary_points = [
    "✓ Zero-shot: Model làm việc mà không có ví dụ",
    "✓ Few-shot: Model học từ một vài ví dụ trong prompt",
    "✓ Chain-of-thought: Hướng dẫn model suy nghĩ từng bước",
    "✓ Task adaptation: Dạy model làm các task mới qua ví dụ",
    "✓ Multi-task: Một prompt có thể dạy nhiều task cùng lúc",
    "✓ Không cần training lại model - chỉ cần thay đổi prompt"
]

for point in summary_points:
    print(point)

print("\n" + "="*80)
print("Demo hoàn thành! In-context learning là khả năng mạnh mẽ của LLM.")
print("Bằng cách cung cấp ví dụ trong prompt, chúng ta có thể dạy model")
print("thực hiện các task mới mà không cần training lại.")
