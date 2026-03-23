cd D:\ECT\Project\FastAPITest
.\.venv\Scripts\activate
uvicorn main:app --host 0.0.0.0
cloudflared tunnel run FastAPI-RAG
