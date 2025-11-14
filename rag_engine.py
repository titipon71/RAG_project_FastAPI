# rag_engine.py
import os
import re
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter


import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb.config import Settings

load_dotenv()

DATA_DIR = os.getenv("RAG_DATA_DIR", "uploads")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# 1) โหลดเอกสาร
try:
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
except ValueError:
    documents = []


for d in documents:
    d.metadata = d.metadata or {}
    d.metadata.setdefault("channel_id", "global")

# 2) embedder
embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL,
    device="cuda",        
    trust_remote_code=True
)

SAFETY_SYSTEM_PROMPT = (
    "คุณคือผู้ช่วยอัจฉริยะที่ตอบคำถามเป็นภาษาไทยอย่างสุภาพและชัดเจน "
    "โปรดตอบเฉพาะคำตอบสุดท้ายเท่านั้น ห้ามแสดงขั้นตอนการคิด "
    "ห้ามใส่ข้อความในแท็ก <think>...</think> "
    "หากไม่แน่ใจ ให้ตอบอย่างกระชับว่าไม่แน่ใจ อย่าคิดเสียงดังหรืออธิบายกระบวนการคิดของคุณ."
)

# 3) LLM = Ollama local
llm = Ollama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    request_timeout=60.0,
    system_prompt=SAFETY_SYSTEM_PROMPT,
    context_window=8192,
    num_output=1536,
    additional_kwargs={
            "options": {   
                "temperature": 0.3,
                "top_p": 0.8,
            }
        },
)

# 4) Chroma persistent
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
)

COLLECTION = "quickstart2"
try:
    chroma_collection = chroma_client.get_collection(COLLECTION)
except:
    chroma_collection = chroma_client.create_collection(COLLECTION)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 5) สร้าง index
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    storage_context=storage_context,
)

# 6) query engine
query_engine = index.as_query_engine(llm=llm)

def _strip_think(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"(?i)(^|\n)\s*(analysis:|reasoning:|thoughts?:).*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
    return text.strip()

def add_documents(docs):
    """เติมเอกสารใหม่เข้า vector store เดิม"""
    for d in docs:
        d.metadata = d.metadata or {}
    # เทคนิค: สร้าง Index ใหม่จาก docs เหล่านี้ โดยใช้ storage_context เดิม
    # LlamaIndex จะฝัง (embed) แล้วเขียนลงคอลเลกชันเดิมให้
    VectorStoreIndex.from_documents(
        docs,
        embed_model=embed_model,
        storage_context=storage_context,
    )

def rag_query(question: str) -> str:
    resp = query_engine.query(question)
    return str(resp)

def rag_query_with_channel(question: str, channel_id: int) -> str:
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="channel_id", value=str(channel_id))]
    )
    retriever = index.as_retriever(similarity_top_k=5, filters=filters)
    nodes = retriever.retrieve(question)

    if not nodes:
        return "ตอนนี้ยังไม่มีไฟล์ในช่องนี้ที่ผมใช้ตอบได้เลยนะ"

    # สร้าง list ชื่อไฟล์จาก metadata ถ้ามี
    file_names = []
    for n in nodes:
        # ถ้าตอน insert อยากเก็บชื่อไฟล์ด้วยก็เก็บเลย
        if "filename" in n.metadata:
            file_names.append(n.metadata["filename"])

    # ตอบปนไปกับการสรุป
    from llama_index.core.response_synthesizers import get_response_synthesizer
    synth = get_response_synthesizer(llm=llm, response_mode="refine")
    resp = synth.synthesize(question, nodes)

    answer = _strip_think(str(resp))
    
    if file_names:
        return f"{answer} (อ้างอิงจากไฟล์: {', '.join(set(file_names))})"
    return answer

