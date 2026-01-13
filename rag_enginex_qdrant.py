# rag_enginex.py
# ============================================================
#                      IMPORTS
# ============================================================
import os
import re
import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

from dotenv import load_dotenv
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

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

# --- REPLACED: Chroma -> Qdrant ---
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http import models # ใช้สำหรับ Filter เวลาลบไฟล์

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.callbacks import CallbackManager

# ==========================================
# 0. Logging Setup (Custom Colors)
# ==========================================
# ... (ส่วน Logging เหมือนเดิม ไม่ต้องแก้) ...
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
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True, show_path=False)]
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
    
    # --- CHANGED: Qdrant Config ---
    # ถ้าใช้ Docker ให้ใส่ URL เช่น "http://localhost:6333"
    # ถ้าจะ save ลง Disk (Local mode) ให้ใส่ path ที่ต้องการใน path=... แต่แนะนำ HTTP
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333") 
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
# 1. Callback Handler (เหมือนเดิม)
# =========================================
class OllamaTokenHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.latest_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    def on_event_start(self, event_type, payload, event_id, **kwargs): pass  
    def on_event_end(self, event_type, payload, event_id, **kwargs):
        if event_type == CBEventType.LLM:
            response = payload.get(EventPayload.RESPONSE)
            if hasattr(response, "raw") and isinstance(response.raw, dict):
                p_tokens = response.raw.get("prompt_eval_count", 0)
                c_tokens = response.raw.get("eval_count", 0)
                self.latest_usage = {"prompt_tokens": p_tokens, "completion_tokens": c_tokens, "total_tokens": p_tokens + c_tokens}
    def start_trace(self, trace_id=None): pass
    def end_trace(self, trace_id=None, trace_map=None): pass

# ==========================================
# 2. RAG Service Class
# ==========================================
class RAGService:
    def __init__(self):
        self.chat_engines = {}
        
        logger.info("[bold cyan]🚀 Initializing RAG Service (Qdrant Edition)...[/]")
        self._ensure_directories()
        
        self.token_handler = OllamaTokenHandler()
        LlamaSettings.callback_manager = CallbackManager([self.token_handler])
        
        # 1. Init Models
        logger.info(f"Loading Embedding Model: [yellow]{config.EMBED_MODEL_NAME}[/]")
        self.embed_model = self._init_embed_model()
        
        logger.info("Initializing Node Parser...")
        self.node_parser = self._init_node_parser()
        
        logger.info(f"Connecting to Ollama: [yellow]{config.OLLAMA_MODEL}[/]")
        self.llm = self._init_llm()
        
        # 2. Init Database & Storage (Qdrant)
        logger.info(f"Connecting to [bright_blue]Qdrant at {config.QDRANT_URL}...[/]")
        
        # --- CHANGED: Init Qdrant Client ---
        self.client = qdrant_client.QdrantClient(
            url=config.QDRANT_URL,
            # ถ้าต้องการ run local (embedded mode) ให้ใช้: path="./qdrant_data" แทน url
        )
        self.vector_store = QdrantVectorStore(
            client=self.client, 
            collection_name=config.COLLECTION_NAME
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # 3. Load Index
        logger.info("Loading/Creating Vector Index...")
        self.index = self._load_or_create_index()

        logger.info(f"Initializing Redis Chat Store at [italic hot_pink]{config.REDIS_URL}[/]...")
        try:
            self.chat_store = RedisChatStore(redis_url=config.REDIS_URL)
            if self.chat_store._redis_client.ping():
                logger.info("[bold green]✅ Connected to Redis Chat Store successfully.[/]")
            else:
                logger.warning("[bold yellow]⚠️ Failed to connected Redis Chat Store.[/]")
        except Exception as e:
            logger.error(f"[bold red]❌ Failed to connect to Redis:[/]\n{e}")
            raise e
                
        logger.info("[bold green]✅ RAG Service Ready![/]")
        
    def _ensure_directories(self):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        # ไม่ต้องสร้าง folder chroma แล้ว

    def _init_embed_model(self) -> HuggingFaceEmbedding:
        return HuggingFaceEmbedding(
            model_name=config.EMBED_MODEL_NAME,
            device="cuda",
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
        )

    def _load_or_create_index(self) -> VectorStoreIndex:
        try:
            # Qdrant จะเก็บ index ไว้ ถ้า connect ได้มันจะโหลดมาเอง
            return VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embed_model,
            )
        except Exception as e:
            logger.warning(f"Could not load index from vector store: {e}")
            # ถ้าเป็นครั้งแรกอาจจะยังไม่มี collection
            
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
        if not text: return text
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"(?i)(^|\n)\s*(analysis:|reasoning:|thoughts?:).*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
        return text.strip()

    def _get_chat_engine(self, channel_id: str, session_id: Union[str, int, None]):
        user_key = str(session_id) if session_id else "global_guest"
        engine_key = f"{channel_id}_{user_key}"
        
        if engine_key in self.chat_engines:
            return self.chat_engines[engine_key]
            
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="channel_id", value=str(channel_id))]
        )
        
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=3000,
            chat_store=self.chat_store,
            chat_store_key=user_key
        )
        
        new_engine = self.index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            similarity_top_k=config.TOP_K,
            filters=filters,
            llm=self.llm,
            response_mode="compact",
        )
        
        self.chat_engines[engine_key] = new_engine
        return new_engine
    
    # --- Public Methods ---

    def add_documents(self, docs: List[Document]):
        if not docs: return
        logger.info(f"Adding [bold]{len(docs)}[/] documents...")
        for d in docs: d.metadata = d.metadata or {}
        
        nodes = self.node_parser.get_nodes_from_documents(docs)
        self.index.insert_nodes(nodes)
        self.index.storage_context.persist() # Qdrant saves automatically, but keeping for consistency
        logger.info(f"Successfully added [bold green]{len(nodes)}[/] nodes to index.")

    # --- CHANGED: Deletion Logic for Qdrant ---
    def delete_documents_by_metadata(self, metadata: Dict[str, Any]):
        """
        Qdrant ใช้ FilterSelector ในการลบ Point ตามเงื่อนไข Metadata
        """
        logger.info(f"Deleting documents where: [yellow]{metadata}[/]")
        
        # สร้าง Conditions จาก Metadata dict
        conditions = []
        for key, value in metadata.items():
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=str(value))
                )
            )
            
        if not conditions:
            return

        try:
            self.client.delete(
                collection_name=config.COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=conditions
                    )
                )
            )
            logger.info("Delete operation sent to Qdrant.")
        except Exception as e:
            logger.error(f"Failed to delete from Qdrant: {e}")

    def delete_documents_by_file_id(self, files_id: Union[str, int]):
        self.delete_documents_by_metadata({"files_id": str(files_id)})

    def clear_session_history(self, sessions_id: Union[str, int]):
        if not sessions_id: return
        user_key = str(sessions_id)
        logger.info(f"🗑️ Clearing chat history for session: [bold]{user_key}[/]")
        try:
            self.chat_store.delete_messages(user_key)
            logger.info(f"[bold green]✅ Successfully cleared history for {user_key}[/]")
        except Exception as e:
            logger.error(f"[bold red]❌ Failed to clear history for {user_key}:[/]\n{e}")

    async def aquery(self, question: str, channel_id: Union[str, int], sessions_id: Union[str, int, None] = None) -> Dict[str, Any]:
        # (เหมือนเดิม)
        logger.info(f"Querying: [bold cyan]{question}[/] (Channel: {channel_id})")
        self.token_handler.latest_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        chat_engine = self._get_chat_engine(str(channel_id), sessions_id)
        response = await chat_engine.achat(question)
        answer_text = self._strip_think(str(response))
        token_usage = self.token_handler.latest_usage
        
        if not response.source_nodes:
            return {
                "answer": "ตอนนี้ยังไม่มีเอกสารที่ใช้ตอบได้เลยนะ🤔 รบกวนเพิ่มเอกสารก่อนนะคะ😊",
                "usage": token_usage,
                "sources": []
            }

        file_names = {node.node.metadata.get("filename") for node in response.source_nodes if node.node.metadata.get("filename")}
        return {"answer": answer_text, "usage": token_usage, "sources": list(file_names)}

    # --- CHANGED: Debug Logic for Qdrant ---
    def debug_list_docs_by_channel(self, channel_id: int):
        print(f"[DEBUG] Qdrant docs for channel {channel_id}")
        
        # Qdrant ใช้ Scroll เพื่อดึงข้อมูลออกมาดู
        try:
            response = self.client.scroll(
                collection_name=config.COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="channel_id",
                            match=models.MatchValue(value=str(channel_id))
                        )
                    ]
                ),
                limit=100,
                with_payload=True
            )
            
            points = response[0] # scroll returns (points, next_page_offset)
            print(f"  total_chunks_found (limit 100): {len(points)}")

            summary = {}
            for point in points:
                payload = point.payload
                file = payload.get("filename", "unknown")
                # LlamaIndex บางทีเก็บ page_label ไว้ใน metadata หรือ relationships
                # อันนี้ดึงคร่าวๆ เท่าที่มี
                summary.setdefault(file, 0)
                summary[file] += 1

            for file, count in summary.items():
                print(f"  - {file}: {count} chunks")
                
        except Exception as e:
            print(f"[DEBUG] Error scrolling Qdrant: {e}")

# ==========================================
# 3. Global Instance
# ==========================================
try:
    rag_engine = RAGService()
except Exception as e:
    logger.exception("Failed to initialize RAG Engine")
    rag_engine = None