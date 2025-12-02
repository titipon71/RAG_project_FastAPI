import os
import re
import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

from dotenv import load_dotenv

# --- LlamaIndex Core ---
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Document,
    Settings as LlamaSettings
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.schema import NodeWithScore

# --- Integrations ---
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAG_ENGINE")

load_dotenv()

# ==========================================
# 1. Configuration
# ==========================================
@dataclass
class AppConfig:
    # Paths
    DATA_DIR: str = os.getenv("RAG_DATA_DIR", "uploads")
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./chroma_db")
    COLLECTION_NAME: str = "quickstart2"
    
    # Models
    EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "scb10x/llama3.2-typhoon2-1b-instruct:latest")
    
    # Parameters
    CONTEXT_WINDOW: int = 4096
    NUM_OUTPUT: int = 512
    TOP_K: int = 3
    CHUNK_SIZE: int = 512 # สำหรับ Splitting
    
    # Prompts
    SAFETY_SYSTEM_PROMPT: str = (
        "คุณคือผู้ช่วยอัจฉริยะที่ตอบคำถามเป็นภาษาไทยอย่างสุภาพและชัดเจน "
        "โปรดตอบเฉพาะคำตอบสุดท้ายเท่านั้น ห้ามแสดงขั้นตอนการคิด "
        "ห้ามใส่ข้อความในแท็ก <think>...</think> "
        "อย่าคิดเสียงดังหรืออธิบายกระบวนการคิดของคุณ."
        "เอาความรู้มาจากเอกสารที่มีเท่านั้น ถ้าไม่มีในเอกสารให้บอกว่า 'ขออภัย ฉันไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่มีอยู่ 😔' "
    )

config = AppConfig()

# ==========================================
# 2. RAG Service Class (Logic หลัก)
# ==========================================
class RAGService:
    def __init__(self):
        logger.info("🚀 Initializing RAG Service...")
        self._ensure_directories()
        
        # 1. Init Models (Heavy Load)
        logger.info(f"Loading Embedding Model: {config.EMBED_MODEL_NAME}")
        self.embed_model = self._init_embed_model()
        
        logger.info("Initializing Node Parser...")
        self.node_parser = self._init_node_parser()
        
        logger.info(f"Connecting to Ollama: {config.OLLAMA_MODEL}")
        self.llm = self._init_llm()
        
        # 2. Init Database & Storage
        logger.info("Connecting to ChromaDB...")
        self.chroma_collection = self._init_chroma()
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # 3. Load Index
        logger.info("Loading/Creating Vector Index...")
        self.index = self._load_or_create_index()
        logger.info("✅ RAG Service Ready!")

    def _ensure_directories(self):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        os.makedirs(config.CHROMA_DIR, exist_ok=True)

    def _init_embed_model(self) -> HuggingFaceEmbedding:
        return HuggingFaceEmbedding(
            model_name=config.EMBED_MODEL_NAME,
            device="cuda", # เปลี่ยนเป็น "cpu" ได้ถ้าเครื่องไม่มีการ์ดจอ
            trust_remote_code=True
        )

    def _init_node_parser(self) -> SemanticSplitterNodeParser:
        return SemanticSplitterNodeParser.from_defaults(
            embed_model=self.embed_model,
            breakpoint_percentile_threshold=95,
            buffer_size=3
        )

    def _init_llm(self) -> Ollama:
        return Ollama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            request_timeout=60.0,
            system_prompt=config.SAFETY_SYSTEM_PROMPT,
            context_window=config.CONTEXT_WINDOW,
            num_output=config.NUM_OUTPUT,
            additional_kwargs={
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.8,
                }
            },
        )

    def _init_chroma(self):
        client = chromadb.PersistentClient(
            path=config.CHROMA_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        return client.get_or_create_collection(config.COLLECTION_NAME)

    def _load_or_create_index(self) -> VectorStoreIndex:
        try:
            # พยายามโหลดจาก DB เดิมก่อน (เร็ว)
            return VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embed_model,
            )
        except Exception as e:
            logger.warning(f"Could not load index from vector store: {e}")
            logger.info("Creating new index from documents in DATA_DIR...")
            
            documents = []
            try:
                documents = SimpleDirectoryReader(config.DATA_DIR).load_data()
                for d in documents:
                    d.metadata = d.metadata or {}
                    d.metadata.setdefault("channel_id", "global")
            except ValueError:
                logger.warning("No documents found to initialize.")

            nodes = self.node_parser.get_nodes_from_documents(documents) if documents else []
            
            return VectorStoreIndex(
                nodes,
                storage_context=self.storage_context,
                embed_model=self.embed_model
            )

    def _strip_think(self, text: str) -> str:
        """ลบข้อความส่วนที่ AI คิด (Chain of Thought) ออก"""
        if not text:
            return text
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"(?i)(^|\n)\s*(analysis:|reasoning:|thoughts?:).*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
        return text.strip()

    # --- Public Methods ---

    def add_documents(self, docs: List[Document]):
        """เพิ่มเอกสารใหม่ลงใน Index เดิม"""
        if not docs:
            return
            
        logger.info(f"Adding {len(docs)} documents...")
        for d in docs:
            d.metadata = d.metadata or {}
        
        # แปลงเป็น Nodes
        nodes = self.node_parser.get_nodes_from_documents(docs)
        
        # Insert ลง Index (สำคัญ: ต้องใช้ insert_nodes ไม่ใช่สร้าง VectorStoreIndex ใหม่)
        self.index.insert_nodes(nodes)
        
        # Persist ลง Storage (ปกติ ChromaVectorStore จะ Auto-persist แต่เรียกไว้เพื่อความชัวร์ในบาง version)
        self.index.storage_context.persist()
        logger.info(f"Successfully added {len(nodes)} nodes to index.")

    def delete_documents_by_metadata(self, metadata: Dict[str, Any]):
        where = {k: str(v) for k, v in metadata.items()}
        logger.info(f"Deleting documents where: {where}")
        self.chroma_collection.delete(where=where)

    def delete_documents_by_file_id(self, files_id: Union[str, int]):
        self.delete_documents_by_metadata({"files_id": str(files_id)})

    def query(self, question: str, channel_id: Union[str, int], sessions_id: Optional[int] = None) -> str:
        logger.info(f"Querying: {question} (Channel: {channel_id})")
        
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="channel_id", value=str(channel_id))]
        )
        
        query_engine = self.index.as_query_engine(
            similarity_top_k=config.TOP_K,
            filters=filters,
            llm=self.llm,                
            response_mode="compact"
        )

        response = query_engine.query(question)

        if not response.source_nodes:
            logger.info("No source nodes found.")
            return "ตอนนี้ยังไม่มีเอกสารที่ใช้ตอบได้เลยนะ🤔 รบกวนเพิ่มเอกสารก่อนนะคะ😊"


        file_names = {
            node.node.metadata.get("filename") 
            for node in response.source_nodes 
            if node.node.metadata.get("filename")
        }
        if file_names:
            logger.info(f"Sources used: {file_names}")

        return self._strip_think(str(response))

    def debug_list_docs(self, channel_id: Union[str, int]):
        res = self.chroma_collection.get(where={"channel_id": str(channel_id)})
        print(f"--- DEBUG: Docs for channel {channel_id} ---")
        print(f"IDs: {res.get('ids')}")
        print(f"Metadatas: {res.get('metadatas')}")
        print("------------------------------------------")


# ==========================================
# 3. Global Instance (Eager Loading)
# ==========================================
# การประกาศตรงนี้จะทำให้ Model โหลดทันทีที่ import ไฟล์นี้
# ข้อดี: เรียกใช้ครั้งแรกเร็ว
# ข้อเสีย: กิน Resource ทันที, start server ช้านิดนึง

try:
    rag_engine = RAGService()
except Exception as e:
    logger.error(f"Failed to initialize RAG Engine: {e}")
    rag_engine = None

# ==========================================
# 4. Helper Functions (Backward Compatibility)
# ==========================================
# ฟังก์ชันพวกนี้มีไว้เพื่อให้โค้ดเก่าของคุณเรียกใช้ได้โดยไม่ต้องแก้เยอะ
# แต่จริงๆ แล้วควรเรียก rag_engine.query() โดยตรงจะดีกว่า

def rag_query_with_channel(question: str, channel_id: int, sessions_id: int) -> str:
    if rag_engine:
        return rag_engine.query(question, channel_id, sessions_id)
    return "System Error: RAG Engine not initialized."

def add_documents(docs):
    if rag_engine:
        rag_engine.add_documents(docs)

def delete_documents_by_file_id(files_id):
    if rag_engine:
        rag_engine.delete_documents_by_file_id(files_id)

def debug_list_docs_by_channel(channel_id):
    if rag_engine:
        rag_engine.debug_list_docs(channel_id)