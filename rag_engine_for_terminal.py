# rag_engine.py
import os
import re
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.node_parser import SemanticSplitterNodeParser


import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb.config import Settings

load_dotenv()

DATA_DIR = os.getenv("RAG_DATA_DIR", "uploads")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "scb10x/llama3.2-typhoon2-1b-instruct:latest")

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

# 2.1) Semantic splitter (chunk แบบ semantic)
node_parser = SemanticSplitterNodeParser.from_defaults(
    embed_model=embed_model,
    breakpoint_percentile_threshold=95,
    buffer_size=3
)

SAFETY_SYSTEM_PROMPT = (
    "คุณคือผู้ช่วยอัจฉริยะที่ตอบคำถามเป็นภาษาไทยอย่างสุภาพและชัดเจน "
    "โปรดตอบเฉพาะคำตอบสุดท้ายเท่านั้น ห้ามแสดงขั้นตอนการคิด "
    "ห้ามใส่ข้อความในแท็ก <think>...</think> "
    "อย่าคิดเสียงดังหรืออธิบายกระบวนการคิดของคุณ."
    "เอาความรู้มาจากเอกสารที่มีเท่านั้น ถ้าไม่มีในเอกสารให้บอกว่า 'ขออภัย ฉันไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่มีอยู่ 😔' "
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

# 5) สร้าง index ด้วย semantic chunks
if documents:
    nodes = node_parser.get_nodes_from_documents(documents)
else:
    nodes = []

index = VectorStoreIndex.from_documents(
    nodes,
    embed_model=embed_model,
    storage_context=storage_context,
)

# 6) query engine
query_engine = index.as_query_engine(llm=llm)


