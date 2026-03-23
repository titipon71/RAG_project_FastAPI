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
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.storage.chat_store.redis import RedisChatStore

from llama_index.core.memory import ChatMemoryBuffer
# --- Integrations ---
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.lancedb import LanceDBVectorStore 
import lancedb                                                      

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.callbacks import CallbackManager

from concurrent.futures import ThreadPoolExecutor

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.chat_engine import CondensePlusContextChatEngine

from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

from threading import RLock

import torch
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
    LANCEDB_DIR: str = os.getenv("LANCEDB_DIR", "./lancedb")     
    TABLE_NAME: str = "quickstart2"                                   
    
    # Models
    EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "ministral-3:3b")
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Parameters
    CONTEXT_WINDOW: int = 4096
    NUM_OUTPUT: int = 512
    TOP_K: int = 5
    CHUNK_SIZE: int = 512 # สำหรับ Splitting
    
    # Prompts
    SAFETY_SYSTEM_PROMPT: str = (
        "คุณคือผู้ช่วยอัจฉริยะที่ตอบคำถามเป็นภาษาไทยอย่างสุภาพและชัดเจน "
        "โปรดตอบเฉพาะคำตอบสุดท้ายเท่านั้น ห้ามแสดงขั้นตอนการคิด "
        "ห้ามใส่ข้อความในแท็ก <think>...</think> "
        "อย่าคิดเสียงดังหรืออธิบายกระบวนการคิดของคุณ."
        "เอาความรู้มาจากเอกสารที่มีเท่านั้น ถ้าไม่มีในเอกสารให้บอกว่า 'ขออภัย ฉันไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่มีอยู่ 😔' "
    )
    
    MAX_CACHE_NODES: int = int(os.getenv("MAX_CACHE_NODES", "100000"))

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
        self.chat_engines = {}
        self.nodes_cache: Dict[str, TextNode] = {}
        self.engine_lock = RLock()
        self.nodes_lock = RLock()
        
        logger.info("[bold cyan]🚀 Initializing RAG Service...[/]")
        self._ensure_directories()

        self.token_handler = OllamaTokenHandler()
        LlamaSettings.callback_manager = CallbackManager([self.token_handler])

        # โหลด EmbedModel และ LLM พร้อมกัน แล้วค่อยสร้าง NodeParser
        logger.info("Loading models in parallel...")
        self._init_models_parallel_sync()

        # ส่วนที่เหลือเหมือนเดิม
        logger.info("Connecting to [bright_blue]LanceDB...[/]")
        self.lance_db, self.vector_store = self._init_lancedb()
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        logger.info("Loading/Creating Vector Index...")
        self.index = self._load_or_create_index()

        logger.info(f"Initializing Redis Chat Store at [italic hot_pink]{config.REDIS_URL}[/]...")
        try:
            self.chat_store = RedisChatStore(redis_url=config.REDIS_URL)
            if self.chat_store._redis_client.ping():
                logger.info("[bold green]✅ Connected to Redis Chat Store successfully.[/]")
            else:
                logger.warning("[bold yellow]⚠️ Failed to connect Redis Chat Store.[/]")
        except Exception as e:
            logger.error(f"[bold red]❌ Failed to connect to Redis:[/]\n{e}")
            raise e

        logger.info("[bold green]✅ RAG Service Ready![/]")

    def _add_nodes_to_cache(self, nodes):

        if not nodes:
            return

        with self.nodes_lock:

            for node in nodes:

                node_id = node.node_id or node.id_

                if not node_id:
                    continue

                self.nodes_cache[node_id] = node

            # enforce size limit
            if len(self.nodes_cache) > config.MAX_CACHE_NODES:

                overflow = len(self.nodes_cache) - config.MAX_CACHE_NODES

                # remove oldest
                keys_to_remove = list(self.nodes_cache.keys())[:overflow]

                for k in keys_to_remove:
                    del self.nodes_cache[k]

            logger.info(
                f"nodes_cache size: {len(self.nodes_cache)}"
            )
    
    def _init_models_parallel_sync(self):
        logger.info(
            f"Loading in parallel — "
            f"EmbedModel: [yellow]{config.EMBED_MODEL_NAME}[/]  |  "
            f"LLM: [yellow]{config.OLLAMA_MODEL}[/]  |  "
            f"Reranker: [yellow]bge-reranker-v2-m3[/]"
        )

        with ThreadPoolExecutor(max_workers=3) as pool:  # เพิ่มเป็น 3
            embed_future    = pool.submit(self._init_embed_model)
            llm_future      = pool.submit(self._init_llm)
            reranker_future = pool.submit(self._init_reranker)  # ← เพิ่ม

            self.embed_model = embed_future.result()
            self.llm         = llm_future.result()
            self.reranker    = reranker_future.result()         # ← เพิ่ม

        logger.info("Initializing Node Parser...")
        self.node_parser = self._init_node_parser()
        
    def _ensure_directories(self):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        os.makedirs(config.LANCEDB_DIR, exist_ok=True) 

    def _init_embed_model(self) -> HuggingFaceEmbedding:
        return HuggingFaceEmbedding(
            model_name=config.EMBED_MODEL_NAME,
            device = "cuda" if torch.cuda.is_available() else "cpu" , # เปลี่ยนเป็น "cpu" ได้ถ้าเครื่องไม่มีการ์ดจอ
            trust_remote_code=True
        )
    
    def _init_reranker(self) -> FlagEmbeddingReranker:
        return FlagEmbeddingReranker(
            model="BAAI/bge-reranker-v2-m3",  # รองรับภาษาไทยได้ดี
            top_n=config.TOP_K,               # เหลือ TOP_K หลัง rerank
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

    def _init_lancedb(self):
        """เชื่อมต่อ LanceDB และสร้าง/โหลด Table"""
        db = lancedb.connect(config.LANCEDB_DIR)
        vector_store = LanceDBVectorStore(
            uri=config.LANCEDB_DIR,
            table_name=config.TABLE_NAME,
        )
        return db, vector_store

    def _load_or_create_index(self) -> VectorStoreIndex:
        try:
            existing_tables = self.lance_db.table_names()
            if config.TABLE_NAME in existing_tables:
                logger.info(f"Found existing LanceDB table: [green]{config.TABLE_NAME}[/]")
                index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                    embed_model=self.embed_model,
                )
                # โหลด nodes เข้า cache หลัง index พร้อม
                self._load_nodes_cache_from_lancedb()
                return index
            else:
                raise ValueError(f"Table '{config.TABLE_NAME}' not found, will create new index.")
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

            # เก็บลง cache ด้วยถ้ามี nodes
            if nodes:
                self._add_nodes_to_cache(nodes)

            return VectorStoreIndex(
                nodes,
                storage_context=self.storage_context,
                embed_model=self.embed_model
            )
    
    def _load_nodes_cache_from_lancedb(self):
        """โหลด nodes จาก LanceDB เข้า nodes_cache สำหรับ BM25"""
        try:
            table = self.lance_db.open_table(config.TABLE_NAME)
            rows = (
                    table
                    .search()
                    .limit(config.MAX_CACHE_NODES)
                    .to_pandas()
                )

            if rows.empty:
                logger.info("LanceDB table is empty, nodes_cache will be empty.")
                return

            from llama_index.core.schema import TextNode

            for _, row in rows.iterrows():
                metadata = row.get("metadata", {})
                # LanceDB เก็บ metadata เป็น dict อยู่แล้ว
                if isinstance(metadata, str):
                    import json
                    metadata = json.loads(metadata)

                node = TextNode(
                    text=row.get("text", ""),
                    metadata=metadata,
                    id_=row.get("id", None),
                )
                self._add_nodes_to_cache([node])

            logger.info(f"Loaded [bold green]{len(self.nodes_cache)}[/] nodes into BM25 cache from LanceDB.")

        except Exception as e:
            logger.warning(f"[yellow]Could not load nodes cache from LanceDB: {e}[/]")
            logger.warning("[yellow]BM25 will start empty and fill up as documents are added.[/]")
            # ไม่ raise — fallback gracefully ให้ BM25 เริ่มว่างได้

    def _strip_think(self, text: str) -> str:
        """ลบข้อความส่วนที่ AI คิด (Chain of Thought) ออก"""
        if not text:
            return text
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"(?i)(^|\n)\s*(analysis:|reasoning:|thoughts?:).*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
        return text.strip()

    def _get_chat_engine(self, channel_id: str, session_id: Union[str, int, None]):
        user_key = str(session_id) if session_id else "global_guest"
        engine_key = f"{channel_id}_{user_key}"
        
        with self.engine_lock:
            if engine_key in self.chat_engines:
                return self.chat_engines[engine_key]

        logger.info(f"Creating new ChatEngine for key: {engine_key}")

        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="channel_id", value=str(channel_id))]
        )

        # --- Vector retriever ---
        vector_retriever = self.index.as_retriever(
            similarity_top_k=config.TOP_K * 2,  # ดึงเผื่อไว้ก่อน fusion จะเหลือ TOP_K
            filters=filters,
        )

        # --- BM25 retriever (filter เฉพาะ nodes ของ channel นี้) ---
        with self.nodes_lock:

            channel_nodes = [
                n
                for n in self.nodes_cache.values()
                if str(n.metadata.get("channel_id", "")) == str(channel_id)
            ]

        if channel_nodes:
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=channel_nodes,
                similarity_top_k=config.TOP_K * 2,
            )
            retrievers = [vector_retriever, bm25_retriever]
            logger.info(f"Using hybrid retriever ({len(channel_nodes)} nodes in BM25)")
        else:
            # ยังไม่มี nodes ใน cache → fallback ใช้ vector อย่างเดียว
            retrievers = [vector_retriever]
            logger.info("No nodes in cache yet, using vector-only retriever")

        # --- Fusion (RRF merge) ---
        hybrid_retriever = QueryFusionRetriever(
            retrievers=retrievers,
            similarity_top_k=config.TOP_K,
            num_queries=1,            # ไม่ generate query เพิ่ม ใช้คำถามเดิม
            mode="reciprocal_rerank", # RRF
            use_async=True,
        )

        memory = ChatMemoryBuffer.from_defaults(
            token_limit=3000,
            chat_store=self.chat_store,
            chat_store_key=user_key
        )

        new_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=hybrid_retriever,
            memory=memory,
            llm=self.llm,
            node_postprocessors=[self.reranker],
            system_prompt=config.SAFETY_SYSTEM_PROMPT,
        )

        with self.engine_lock:
            existing_engine = self.chat_engines.get(engine_key)
            if existing_engine is not None:
                return existing_engine
            self.chat_engines[engine_key] = new_engine

        return new_engine
    
    # --- Public Methods ---

    def add_documents(self, docs: List[Document]):
        if not docs:
            return

        logger.info(f"Adding [bold]{len(docs)}[/] documents...")
        for d in docs:
            d.metadata = d.metadata or {}

        nodes = self.node_parser.get_nodes_from_documents(docs)
        self.index.insert_nodes(nodes)
        self._add_nodes_to_cache(nodes)

        # ← clear เพื่อให้ engine ถัดไปสร้าง BM25 ใหม่จาก nodes ที่อัปเดตแล้ว
        with self.engine_lock:
            self.chat_engines.clear()
            
        logger.info(f"Successfully added [bold green]{len(nodes)}[/] nodes to index.")

    def delete_documents_by_metadata(self, metadata: Dict[str, Any]):
        where_clauses = [f"{k} = '{v}'" for k, v in metadata.items()]
        where_str = " AND ".join(where_clauses)
        logger.info(f"Deleting documents where: [yellow]{where_str}[/]")

        try:
            table = self.lance_db.open_table(config.TABLE_NAME)
            table.delete(where_str)

            # ← sync nodes_cache ด้วย
            with self.nodes_lock:

                before = len(self.nodes_cache)

                keys_to_delete = [
                    node_id
                    for node_id, node in self.nodes_cache.items()
                    if all(
                        str(node.metadata.get(k, "")) == str(v)
                        for k, v in metadata.items()
                    )
                ]

                for k in keys_to_delete:
                    del self.nodes_cache[k]

                removed = before - len(self.nodes_cache)
            logger.info(f"[bold green]✅ Delete successful. Removed {removed} nodes from cache.[/]")

            # chat_engines ที่ build ไปแล้วอาจใช้ BM25 เก่า → clear cache
            with self.engine_lock:
                self.chat_engines.clear()
                
            logger.info("Chat engine cache cleared after document deletion.")

        except Exception as e:
            logger.error(f"[bold red]❌ Failed to delete documents:[/]\n{e}")

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

    
    async def aquery(self, question: str, channel_id: Union[str, int], sessions_id: Union[str, int, None] = None) -> Dict[str, Any]:
            logger.info(f"Querying: [bold cyan]{question}[/] (Channel: {channel_id})")
            self.token_handler.latest_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            chat_engine = self._get_chat_engine(str(channel_id), sessions_id)

            response = await chat_engine.achat(question)
            
            answer_text = self._strip_think(str(response))
            token_usage = self.token_handler.latest_usage
            
            if not response.source_nodes:
                logger.info("[yellow]No source nodes found.[/]")
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

            return {
                "answer": answer_text,
                "usage": token_usage,
                "sources": list(file_names)
            }
            
    def debug_list_docs_by_channel(self, channel_id: int):
        """แสดงรายการเอกสารใน LanceDB สำหรับ channel ที่กำหนด"""
        try:
            table = self.lance_db.open_table(config.TABLE_NAME)
            # LanceDB ใช้ SQL filter แทน where dict ของ ChromaDB
            results = table.search().where(f"channel_id = '{channel_id}'").to_list()
            
            print(f"[DEBUG] LanceDB docs for channel {channel_id}")
            print(f"  total_docs: {len(results)}")

            summary = {}
            for row in results:
                metadata = row.get("metadata", {})
                file = metadata.get("file_name", "unknown")
                page = metadata.get("page_label", "?")
                summary.setdefault(file, set()).add(page)

            for file, pages in summary.items():
                pages = sorted(pages)
                print(f"  - {file}: pages={pages}")
        except Exception as e:
            logger.error(f"[bold red]❌ debug_list_docs_by_channel failed:[/]\n{e}")


# ==========================================
# 3. Global Instance (Eager Loading)
# ==========================================

try:
    rag_engine = RAGService()
except Exception as e:
    logger.exception("Failed to initialize RAG Engine")
    rag_engine = None