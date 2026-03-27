import requests
import json
import time

def test_ollama(model_name="gemma3:1b"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": "เล่าเรื่องสั้นเกี่ยวกับแมวอวกาศ 1 ประโยค",
        "stream": False
    }

    print(f"🚀 เริ่มการทดสอบโมเดล: {model_name}...")
    
    start_time = time.time()
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        end_time = time.time()
        
        data = response.json()
        duration = end_time - start_time
        
        print("-" * 30)
        print(f"📄 คำตอบ: {data['response']}")
        print("-" * 30)
        print(f"⏱️ เวลาที่ใช้: {duration:.2f} วินาที")
        
        # คำนวณความเร็วคร่าวๆ
        if 'eval_count' in data:
            tps = data['eval_count'] / (data['eval_duration'] / 1e9)
            print(f"⚡ ความเร็ว: {tps:.2f} tokens/s")
            
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    test_ollama()