import requests
import json
import time

def test_ollama_stream(model_name="gemma3:1b"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": "เล่าเรื่องสั้นเกี่ยวกับแมวอวกาศ 1 ประโยค",
        "stream": True
    }

    print(f"🚀 เริ่มการทดสอบโมเดล (streaming): {model_name}...\n")
    
    start_time = time.time()
    
    try:
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            
            full_response = ""
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    
                    # ดึงข้อความทีละ chunk
                    chunk = data.get("response", "")
                    print(chunk, end="", flush=True)
                    full_response += chunk
                    
                    # เช็คว่า stream จบหรือยัง
                    if data.get("done", False):
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        print("\n" + "-" * 30)
                        print(f"⏱️ เวลาที่ใช้: {duration:.2f} วินาที")
                        
                        if 'eval_count' in data and 'eval_duration' in data:
                            tps = data['eval_count'] / (data['eval_duration'] / 1e9)
                            print(f"⚡ ความเร็ว: {tps:.2f} tokens/s")

    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    test_ollama_stream()