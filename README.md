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
* **Vector Database:** ChromaDB / FAISS / Qdrant
* **LLM Provider:** Ollama

## 📂 โครงสร้างโปรเจกต์ (Project Structure)

```
FastAPITest/
├── main.py                  # Entry point ของแอป FastAPI (รวม Router, Middleware, Static Files)
├── requirements.txt         # รายการ Python packages ที่จำเป็น
├── rag.sql                  # SQL schema สำหรับสร้างฐานข้อมูล
├── rag_enginex.py           # RAG engine หลักสำหรับ Retrieval-Augmented Generation
├── rag_enginex_qdrant.py    # RAG engine เวอร์ชันที่ใช้ Qdrant เป็น Vector DB
├── rag_engine_old_version.py# RAG engine เวอร์ชันเก่า
├── rag_engine_for_terminal.py# RAG engine สำหรับใช้งานผ่าน Terminal
├── Hashids_demo.py          # สคริปต์ตัวอย่างการใช้งาน Hashids
│
├── core/                    # ⚙️ การตั้งค่าหลักและ Utility ของระบบ
│   ├── config.py            #   ค่า Settings ต่าง ๆ (Database URL, Secret Key, SSO, etc.)
│   ├── cors.py              #   กำหนดค่า CORS (Cross-Origin Resource Sharing)
│   ├── security.py          #   ระบบ Authentication (JWT Token, Password Hashing)
│   ├── api_key_security.py  #   ระบบยืนยันตัวตนผ่าน API Key
│   ├── hashids.py           #   Encode/Decode ID ด้วย Hashids
│   ├── enums.py             #   ค่าคงที่แบบ Enum ที่ใช้ในระบบ
│   ├── logging.py           #   ตั้งค่า Logging
│   └── tag.py               #   Metadata ของ API Tags สำหรับเอกสาร Swagger
│
├── db/                      # 🗄️ เลเยอร์ฐานข้อมูล (Database Layer)
│   ├── base.py              #   Base class ของ SQLAlchemy ORM
│   ├── session.py           #   สร้าง Database Session / Engine
│   └── models/              #   📋 ORM Models (ตารางในฐานข้อมูล)
│       ├── user.py           #     โมเดลผู้ใช้งาน
│       ├── channel.py        #     โมเดลช่องแชท
│       ├── chat.py           #     โมเดลข้อความแชท
│       ├── session.py        #     โมเดล Session การสนทนา
│       ├── event.py          #     โมเดลเหตุการณ์/กิจกรรม
│       ├── file.py           #     โมเดลไฟล์ที่อัปโหลด
│       └── api_key.py        #     โมเดล API Key
│
├── schemas/                 # 📐 Pydantic Schemas (Request/Response Validation)
│   ├── base.py              #   Base schema ที่ใช้ร่วมกัน
│   ├── auth.py              #   Schema สำหรับ Login/Register
│   ├── user.py              #   Schema ข้อมูลผู้ใช้
│   ├── channel.py           #   Schema ช่องแชท
│   ├── chat.py              #   Schema ข้อความแชท
│   ├── session.py           #   Schema Session
│   ├── event.py             #   Schema เหตุการณ์
│   ├── file.py              #   Schema ไฟล์
│   ├── api_key.py           #   Schema API Key
│   └── moderation.py        #   Schema สำหรับระบบ Moderation
│
├── routers/                 # 🌐 API Endpoints (Route Handlers)
│   ├── auth.py              #   เส้นทาง Authentication (Login, Register, SSO)
│   ├── users.py             #   เส้นทางจัดการผู้ใช้
│   ├── channels.py          #   เส้นทางจัดการช่องแชท
│   ├── files.py             #   เส้นทางอัปโหลด/จัดการไฟล์
│   ├── session.py           #   เส้นทางจัดการ Session
│   ├── events.py            #   เส้นทางจัดการเหตุการณ์
│   ├── statistics.py        #   เส้นทางดูสถิติ
│   ├── api_key.py           #   เส้นทางจัดการ API Key
│   └── utility.py           #   เส้นทาง Utility ทั่วไป
│
├── services/                # 🔧 Business Logic Layer
│   ├── auth_service.py      #   Logic การ Authentication
│   ├── user_service.py      #   Logic จัดการผู้ใช้
│   ├── channel_service.py   #   Logic จัดการช่องแชท
│   ├── file_service.py      #   Logic จัดการไฟล์
│   ├── session_service.py   #   Logic จัดการ Session
│   └── rag_service.py       #   Logic ระบบ RAG (ค้นหาและตอบคำถามจากเอกสาร)
│
├── templates/               # 🖼️ HTML Templates (Jinja2)
│   ├── index.html           #   หน้าแรกของเว็บ
│   └── hashids_demo.html    #   หน้าสาธิตการใช้ Hashids
│
├── file_storage/            # 📁 พื้นที่จัดเก็บไฟล์
│   ├── uploads/             #   ไฟล์ที่อัปโหลดโดยผู้ใช้
│   └── trash/               #   ไฟล์ที่ถูกลบ (Soft Delete)
│
├── chroma_db/               # 🧠 ChromaDB Vector Database
│   └── chroma.sqlite3       #   ไฟล์ฐานข้อมูล Vector สำหรับ RAG
│
└── uploads/                 # 📤 โฟลเดอร์อัปโหลดเพิ่มเติม
```

### สรุปหน้าที่แต่ละ Layer

| Layer | โฟลเดอร์ | หน้าที่ |
|-------|----------|---------|
| **Core** | `core/` | การตั้งค่า, ความปลอดภัย, Utility กลางของระบบ |
| **Database** | `db/` | เชื่อมต่อฐานข้อมูล และกำหนดโครงสร้างตาราง (ORM Models) |
| **Schemas** | `schemas/` | กำหนดรูปแบบ Request/Response เพื่อ Validate ข้อมูลด้วย Pydantic |
| **Routers** | `routers/` | กำหนดเส้นทาง API Endpoints (รับ Request → ส่งต่อ Service) |
| **Services** | `services/` | Business Logic หลัก (ประมวลผล → เรียก DB → ส่งผลลัพธ์กลับ) |
| **Templates** | `templates/` | ไฟล์ HTML สำหรับ Render หน้าเว็บ |
| **Storage** | `file_storage/` | จัดเก็บไฟล์ที่อัปโหลดและไฟล์ที่ถูกลบ |
| **Vector DB** | `chroma_db/` | ฐานข้อมูล Vector สำหรับระบบ RAG (Semantic Search) |

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