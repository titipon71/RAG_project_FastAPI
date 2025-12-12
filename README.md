# RAG Project with FastAPI 🚀

โปรเจกต์นี้คือระบบ Backend API ที่พัฒนาด้วย **FastAPI** โดยประยุกต์ใช้เทคนิค **RAG (Retrieval-Augmented Generation)** เพื่อช่วยให้ AI สามารถค้นหาและตอบคำถามจากข้อมูลเอกสารเฉพาะทาง (Custom Data) ได้อย่างแม่นยำ

## ✨ ฟีเจอร์หลัก (Features)

* **FastAPI Backend:** โครงสร้าง API ที่มีความเร็วสูง รองรับ Asynchronous
* **Document Ingestion:** รองรับการอัปโหลดไฟล์เอกสารเพื่อประมวลผล (PDF, Text, etc.)
* **Vector Search:** ใช้ระบบค้นหาแบบ Semantic Search เพื่อดึงข้อมูลที่เกี่ยวข้อง
* **LLM Integration:** เชื่อมต่อกับ Large Language Model (เช่น OpenAI, Gemini, Local LLM) เพื่อสรุปคำตอบ
* **API Documentation:** มาพร้อม Swagger UI และ ReDoc สำหรับทดสอบระบบได้ทันที

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Web Framework:** FastAPI
* **RAG Framework:** LlamaIndex
* **Vector Database:** ChromaDB / FAISS / Qdrant *(กรุณาตรวจสอบและแก้ไขให้ตรงกับที่คุณใช้)*
* **LLM Provider:** Ollama

## ⚙️ การติดตั้ง (Installation)

1.  **Clone Repository**
    ```bash
    git clone [https://github.com/titipon71/RAG_project_FastAPI.git](https://github.com/titipon71/RAG_project_FastAPI.git)
    cd RAG_project_FastAPI
    ```

2.  **สร้าง Virtual Environment**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **ติดตั้ง Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **ตั้งค่า Environment Variables**
    สร้างไฟล์ `.env` ที่ root folder และกำหนดค่าที่จำเป็น:
    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    # หรือค่า Config อื่นๆ ที่จำเป็นสำหรับ Vector DB หรือ Model
    ```

## 🚀 การใช้งาน (Usage)

เริ่มการทำงานของ Server ด้วยคำสั่ง:

```bash
uvicorn main:app --reload