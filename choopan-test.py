import time
import httpx
from ollama import Client, ResponseError

# ตั้งค่า Client
client = Client(host='http://172.16.212.100:11434', timeout=240.0)

def chat_with_timer():
    print("⏳ กำลังส่งคำถามไปยัง Ollama...")
    
    # เริ่มจับเวลา
    start_time = time.perf_counter()
    
    try:
        response = client.chat(
            model='qwen2.5-coder:latest',
            messages=[{'role': 'user', 'content': 'เรื่องน่ารู้เกี่ยวกับแมว สั้นๆ 1 ประโยค'}],
        )
        
        # คำนวณเวลาที่ใช้
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # แสดงผลลัพธ์
        print(f"\n✅ คำตอบจาก AI:")
        print(response['message']['content'])
        print("-" * 30)
        print(f"⏱️ ใช้เวลาประมวลผลทั้งสิ้น: {execution_time:.2f} วินาที")

    except httpx.ConnectTimeout:
        print("❌ Error: เชื่อมต่อไม่สำเร็จ (Connect Timeout)")
    except httpx.ReadTimeout:
        print("❌ Error: Server ใช้เวลาตอบกลับนานเกินไป (Read Timeout)")
    except ResponseError as e:
        print(f"❌ Ollama Error: {e.error}")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    chat_with_timer()