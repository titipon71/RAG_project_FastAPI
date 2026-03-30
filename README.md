# RAG Project with FastAPI

โปรเจกต์นี้เป็น Backend API ด้วย FastAPI สำหรับงาน RAG (Retrieval-Augmented Generation) รองรับการอัปโหลดเอกสาร, สร้างดัชนีเวกเตอร์, และเรียก LLM เพื่อสรุปหรือถามตอบจากข้อมูลภายในระบบ

## Features

- FastAPI backend (async) พร้อมแยกชั้น routers/services/schemas/db
- รองรับอัปโหลดเอกสารและเก็บไฟล์แบบ soft delete
- RAG engine หลักใช้ LanceDB และมีไฟล์ตัวอย่าง/เวอร์ชัน Qdrant แยกต่างหาก
- รองรับหลาย LLM provider: Ollama, OpenRouter, Google Gemini
- รองรับ OCR สำหรับ PDF ผ่าน EasyOCR + PyMuPDF
- เอกสาร API หลายรูปแบบ: Swagger, ReDoc, Scalar, RapiDoc

## Tech Stack

- Python 3.11+
- FastAPI + Uvicorn
- SQLAlchemy (Async) + MariaDB
- LlamaIndex
- Vector DB: LanceDB (หลัก), Qdrant (ทางเลือก)
- Redis (chat store / memory)
- OCR: EasyOCR, PyMuPDF
- Docker + Docker Compose

## Project Structure

```text
FastAPITest/
├── main.py
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── rag.sql
├── rag_enginex.py
├── rag_enginex_qdrant.py
├── rag_engine_old_version.py
├── rag_engine_for_terminal.py
├── Hashids_demo.py
├── test-gemini.py
├── test-mistral.py
├── test-ocr.py
├── test-ollama.py
│
├── core/
│   ├── __init__.py
│   ├── api_key_security.py
│   ├── config.py
│   ├── cors.py
│   ├── enums.py
│   ├── hashids.py
│   ├── logging.py
│   ├── security.py
│   └── tag.py
│
├── db/
│   ├── __init__.py
│   ├── base.py
│   ├── session.py
│   └── models/
│       ├── __init__.py
│       ├── account_type.py
│       ├── api_key.py
│       ├── channel.py
│       ├── chat.py
│       ├── event.py
│       ├── file.py
│       ├── file_size.py
│       ├── session.py
│       └── user.py
│
├── routers/
│   ├── __init__.py
│   ├── account_type.py
│   ├── api_key.py
│   ├── auth.py
│   ├── channels.py
│   ├── events.py
│   ├── files.py
│   ├── file_size.py
│   ├── session.py
│   ├── statistics.py
│   ├── users.py
│   └── utility.py
│
├── schemas/
│   ├── __init__.py
│   ├── account_types.py
│   ├── api_key.py
│   ├── auth.py
│   ├── base.py
│   ├── channel.py
│   ├── chat.py
│   ├── event.py
│   ├── file.py
│   ├── file_size.py
│   ├── moderation.py
│   ├── session.py
│   └── user.py
│
├── services/
│   ├── __init__.py
│   ├── auth_service.py
│   ├── channel_service.py
│   ├── file_service.py
│   ├── ocr_service.py
│   ├── rag_service.py
│   ├── session_service.py
│   └── user_service.py
│
├── templates/
│   ├── index.html
│   ├── hashids_demo.html
│   ├── callback.html
│   └── callbackicon.png
│
├── file_storage/
│   ├── uploads/
│   └── trash/
│
├── lancedb/
│   └── quickstart2.lance/
│
└── storage/
    ├── docstore.json
    ├── graph_store.json
    ├── image__vector_store.json
    └── index_store.json
```

## Layer Summary

| Layer | Folder | Purpose |
|---|---|---|
| Core | core/ | Config, security, CORS, logging, shared utilities |
| Database | db/ | SQLAlchemy base/session + ORM models |
| Schemas | schemas/ | Pydantic request/response validation |
| Routers | routers/ | API endpoints |
| Services | services/ | Business logic and integration layer |
| Templates | templates/ | HTML templates for web pages/callback |
| File Storage | file_storage/ | Uploaded files and soft-deleted files |
| Vector Storage | lancedb/, storage/ | Vector index and persisted index metadata |

## Installation

1. Clone repository

```bash
git clone https://github.com/titipon71/RAG_project_FastAPI.git
cd RAG_project_FastAPI
```

2. Create virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Create `.env`

ตัวอย่างค่าที่ควรมีอย่างน้อย:

```env
DATABASE_URL=mysql+asyncmy://rag:rag_password@localhost:3306/rag?charset=utf8mb4
SECRET_KEY=change_this_secret
HASH_SALT=change_this_salt

REDIS_URL=redis://localhost:6379
RAG_DATA_DIR=uploads
LANCEDB_DIR=./lancedb

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:1b

# Optional providers
USE_OPENROUTER=false
OPENROUTER_API_KEY=
OPENROUTER_MODEL=mistralai/ministral-3b-2512

USE_GEMINI=false
GEMINI_API_KEY=
GEMINI_MODEL=models/gemini-2.0-flash
```

## Usage

### Run locally

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Run with Docker Compose

```bash
docker compose up -d --build
```

## API Docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Scalar: http://localhost:8000/scalar
- RapiDoc: http://localhost:8000/rapidoc

## Health Check

```bash
curl http://localhost:8000/healthz
```