import os
from mistralai import Mistral

# 1. ตั้งค่า API Key (แนะนำให้ใส่ใน Environment Variable เพื่อความปลอดภัย)
api_key = "ใส่_API_KEY_ของคุณที่นี่" 
model = "mistral-tiny" # หรือใช้ 'mistral-small-latest', 'mistral-medium-latest'

# 2. เริ่มต้น Client
client = Mistral(api_key=api_key)

def test_mistral():
    try:
        print(f"--- กำลังทดสอบ Model: {model} ---")
        
        # 3. ส่งคำถามไปยัง API
        chat_response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "ช่วยแนะนำตัวเองสั้นๆ เป็นภาษาไทยหน่อยครับ",
                },
            ]
        )

        # 4. แสดงผลลัพธ์
        print("\nคำตอบจาก AI:")
        print(chat_response.choices[0].message.content)
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    test_mistral()