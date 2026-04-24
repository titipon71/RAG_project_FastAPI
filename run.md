cd D:\ECT\Project\FastAPITest
.\.venv\Scripts\activate
uvicorn main:app --host 0.0.0.0
cloudflared tunnel run FastAPI-RAG

sudo systemctl restart rag-api

journalctl -u rag-api -f | ccze -A

