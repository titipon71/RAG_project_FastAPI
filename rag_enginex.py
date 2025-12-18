import os
import re
import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

from dotenv import load_dotenv
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from rich.logging import RichHandler

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
from llama_index.storage.chat_store.redis import RedisChatStore

from llama_index.core.memory import ChatMemoryBuffer
# --- Integrations ---
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.callbacks import CallbackManager
# ==========================================
# 0. Logging Setup (Custom Colors)
# ==========================================

custom_theme = Theme({
    "log.time": "bright_white",
    "logging.level.debug": "cyan dim",
    "logging.level.info": "bold #00afff",       
    "logging.level.warning": "bold yellow",
    "logging.level.error": "bold red",
    "logging.level.critical": "bold white on red",    
})

console = Console(theme=custom_theme)

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=console,            
            rich_tracebacks=True,       
            markup=True,                
            show_path=False             
        )
    ]
)

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
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
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
# 1. Callback Handler for Ollama Token Usage
# =========================================

class OllamaTokenHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.latest_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def on_event_start(self, event_type, payload, event_id, **kwargs):
        pass  

    def on_event_end(self, event_type, payload, event_id, **kwargs):
        if event_type == CBEventType.LLM:
            response = payload.get(EventPayload.RESPONSE)
            # ตรวจสอบว่ามีข้อมูล raw จาก Ollama หรือไม่
            if hasattr(response, "raw") and isinstance(response.raw, dict):
                p_tokens = response.raw.get("prompt_eval_count", 0)
                c_tokens = response.raw.get("eval_count", 0)
                
                self.latest_usage = {
                    "prompt_tokens": p_tokens,
                    "completion_tokens": c_tokens,
                    "total_tokens": p_tokens + c_tokens
                }
    
    def start_trace(self, trace_id=None):
        pass

    def end_trace(self, trace_id=None, trace_map=None):
        pass

# ==========================================
# 2. RAG Service Class
# ==========================================
class RAGService:
    def __init__(self):
        # ใส่สี [bold cyan] เพื่อความสวยงาม
        logger.info("[bold cyan]🚀 Initializing RAG Service...[/]")
        self._ensure_directories()
        
        self.token_handler = OllamaTokenHandler()
        LlamaSettings.callback_manager = CallbackManager([self.token_handler])
        
        # 1. Init Models (Heavy Load)
        logger.info(f"Loading Embedding Model: [yellow]{config.EMBED_MODEL_NAME}[/]")
        self.embed_model = self._init_embed_model()
        
        logger.info("Initializing Node Parser...")
        self.node_parser = self._init_node_parser()
        
        logger.info(f"Connecting to Ollama: [yellow]{config.OLLAMA_MODEL}[/]")
        self.llm = self._init_llm()
        
        # 2. Init Database & Storage
        logger.info("Connecting to [bright_blue]ChromaDB...[/]")
        self.chroma_collection = self._init_chroma()
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # 3. Load Index
        logger.info("Loading/Creating Vector Index...")
        self.index = self._load_or_create_index()

        logger.info(f"Initializing Redis Chat Store at [italic hot_pink]{config.REDIS_URL}[/]...")
        try:
            self.chat_store = RedisChatStore(redis_url=config.REDIS_URL)
            # ตรวจสอบว่า Redis ต่อติดจริงไหม
            if self.chat_store._redis_client.ping():
                logger.info("[bold green]✅ Connected to Redis Chat Store successfully.[/]")
            else:
                logger.warning("[bold yellow]⚠️ Failed to connected Redis Chat Store.[/]")
        except Exception as e:
            logger.error(f"[bold red]❌ Failed to connect to Redis:[/]\n{e}")
            # Fallback or raise error depending on requirement
            raise e
                
        logger.info("[bold green]✅ RAG Service Ready![/]")
        
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
            request_timeout=120.0,
            context_window=config.CONTEXT_WINDOW,
            num_output=config.NUM_OUTPUT,
            system_prompt=config.SAFETY_SYSTEM_PROMPT,
            # temperature=0.3,
            # additional_kwargs={
            #         "top_p": 0.8,
            # },
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
            
        logger.info(f"Adding [bold]{len(docs)}[/] documents...")
        for d in docs:
            d.metadata = d.metadata or {}
        
        # แปลงเป็น Nodes
        nodes = self.node_parser.get_nodes_from_documents(docs)
        
        # Insert ลง Index (สำคัญ: ต้องใช้ insert_nodes ไม่ใช่สร้าง VectorStoreIndex ใหม่)
        self.index.insert_nodes(nodes)
        
        # Persist ลง Storage (ปกติ ChromaVectorStore จะ Auto-persist แต่เรียกไว้เพื่อความชัวร์ในบาง version)
        self.index.storage_context.persist()
        logger.info(f"Successfully added [bold green]{len(nodes)}[/] nodes to index.")

    def delete_documents_by_metadata(self, metadata: Dict[str, Any]):
        where = {k: str(v) for k, v in metadata.items()}
        logger.info(f"Deleting documents where: [yellow]{where}[/]")
        self.chroma_collection.delete(where=where)

    def delete_documents_by_file_id(self, files_id: Union[str, int]):
        self.delete_documents_by_metadata({"files_id": str(files_id)})

    def clear_session_history(self, sessions_id: Union[str, int]):

        if not sessions_id:
            logger.warning("⚠️ No session_id provided to clear history.")
            return

        user_key = str(sessions_id)
        logger.info(f"🗑️ Clearing chat history for session: [bold]{user_key}[/]")
        
        try:
            self.chat_store.delete_messages(user_key)
            logger.info(f"[bold green]✅ Successfully cleared history for {user_key}[/]")
        except Exception as e:
            logger.error(f"[bold red]❌ Failed to clear history for {user_key}:[/]\n{e}")

    
    def query(self, question: str, channel_id: Union[str, int], sessions_id: Optional[int] = None) -> Dict[str, Any]:
            logger.info(f"Querying: [bold cyan]{question}[/] (Channel: {channel_id})")
            self.token_handler.latest_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="channel_id", value=str(channel_id))]
            )
            
            user_key = str(sessions_id) if sessions_id else "global_guest"
            
            memory = ChatMemoryBuffer.from_defaults(
                token_limit=3000,
                chat_store=self.chat_store,  
                chat_store_key=user_key     
            )
            
            chat_engine = self.index.as_chat_engine(
                chat_mode="context",
                memory=memory,
                similarity_top_k=config.TOP_K,
                filters=filters,
                llm=self.llm,                
                response_mode="compact",
            )

            response = chat_engine.chat(question)
            
            # 1. ดึงข้อความคำตอบ
            answer_text = self._strip_think(str(response))

            # 2. ดึงจำนวน Token (Ollama ส่งมาใน raw)
            token_usage = self.token_handler.latest_usage
            
            if not response.source_nodes:
                logger.info("[yellow]No source nodes found.[/]")
                # กรณีไม่เจอเอกสาร ก็ return usage เป็น 0 หรือค่าเท่าที่มี
                return {
                    "answer": "ตอนนี้ยังไม่มีเอกสารที่ใช้ตอบได้เลยนะ🤔 รบกวนเพิ่มเอกสารก่อนนะคะ😊",
                    "usage": token_usage,
                    "sources": []
                }

            file_names = {
                node.node.metadata.get("filename") 
                for node in response.source_nodes 
                if node.node.metadata.get("filename")
            }
            if file_names:
                logger.info(f"Sources used: [green]{file_names}[/]")

            # ส่งกลับเป็น Dictionary แทน String
            return {
                "answer": answer_text,
                "usage": token_usage,
                "sources": list(file_names)
            }
            
    def debug_list_docs_by_channel(self, channel_id: int):
        res = self.chroma_collection.get(where={"channel_id": str(channel_id)})

        ids = res.get("ids", [])
        metadatas = res.get("metadatas", [])

        print(f"[DEBUG] chroma docs for channel {channel_id}")
        print(f"  total_docs: {len(ids)}")

        summary = {}
        for m in metadatas:
            file = m.get("file_name", "unknown")
            page = m.get("page_label", "?")

            summary.setdefault(file, set()).add(page)

        for file, pages in summary.items():
            pages = sorted(pages)
            print(f"  - {file}: pages={pages}")



# ==========================================
# 3. Global Instance (Eager Loading)
# ==========================================

try:
    rag_engine = RAGService()
except Exception as e:
    logger.exception("Failed to initialize RAG Engine")
    rag_engine = None