from email.mime import text
import os
import re
import logging
from typing import List, Optional, Dict, Any, Union, AsyncGenerator
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
    Settings as LlamaSettings,
    DocumentSummaryIndex,
    get_response_synthesizer
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.schema import TextNode
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
from pythainlp.tokenize import word_tokenize
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.gemini import Gemini

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
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3:1b")
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Parameters
    CONTEXT_WINDOW: int = 32768
    NUM_OUTPUT: int = 512
    TOP_K: int = 5
    CHUNK_SIZE: int = 512 # สำหรับ Splitting
    
    # Prompts
    SAFETY_SYSTEM_PROMPT: str = (
        "คุณคือผู้ช่วยอัจฉริยะที่ตอบคำถามอย่างสุภาพและชัดเจน "
        "หากผู้ใช้ถามสรุป ให้ลองพิจารณาข้อมูลจากส่วน metadata 'section_summary' หรือภาพรวมของเอกสารที่แนบมาด้วย"
        "เอาความรู้มาจากเอกสารที่มีเท่านั้น ถ้าไม่มีในเอกสารให้บอกว่า 'ขออภัย ฉันไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่มีอยู่ 😔' ถ้ามีก็ไม่เป็นปัญหา"
    )
    
    MAX_CACHE_NODES: int = int(os.getenv("MAX_CACHE_NODES", "100000"))

    
    # === OpenRouter ===
    USE_OPENROUTER: bool = os.getenv("USE_OPENROUTER", "false").lower() == "true"
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "mistralai/ministral-3b-2512")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_CONTEXT_WINDOW: int = int(os.getenv("OPENROUTER_CONTEXT_WINDOW", "262144"))
    
    # === Google Gemini ===
    USE_GEMINI: bool = os.getenv("USE_GEMINI", "false").lower() == "true"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash")
    GEMINI_CONTEXT_WINDOW: int = int(os.getenv("GEMINI_CONTEXT_WINDOW", "1048576"))
    
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

                # Ollama path
                if hasattr(response, "raw") and isinstance(response.raw, dict):
                    p = response.raw.get("prompt_eval_count", 0)
                    c = response.raw.get("eval_count", 0)
                    self.latest_usage = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}

                # OpenAI-compatible path (OpenRouter)
                elif hasattr(response, "raw") and hasattr(response.raw, "usage"):
                    usage = response.raw.usage
                    p = getattr(usage, "prompt_tokens", 0)
                    c = getattr(usage, "completion_tokens", 0)
                    self.latest_usage = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}
    
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
                keys_to_remove = list(self.nodes_cache.keys())[:overflow]
                for k in keys_to_remove:
                    del self.nodes_cache[k]

        # ✅ log แค่ครั้งเดียวตอนเสร็จ
        logger.info(f"nodes_cache size: {len(self.nodes_cache)} (+{len(nodes)} added)")
    
    def _build_summary_nodes(self, docs: List[Document]) -> List[TextNode]:

        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            use_async=False  # แก้ให้ตรงกับ sync context ด้วย
        )

        summary_index = DocumentSummaryIndex.from_documents(
            docs,
            llm=self.llm,
            embed_model=self.embed_model,
            response_synthesizer=response_synthesizer,
            show_progress=True,
        )

        return list(summary_index.docstore.get_all_nodes())
    
    def _init_models_parallel_sync(self):
        if config.USE_GEMINI:
            llm_label = f"Gemini/{config.GEMINI_MODEL}"
        elif config.USE_OPENROUTER:
            llm_label = f"OpenRouter/{config.OPENROUTER_MODEL}"
        else:
            llm_label = f"Ollama/{config.OLLAMA_MODEL}"
            
        logger.info(
            f"Loading in parallel — "
            f"EmbedModel: [yellow]{config.EMBED_MODEL_NAME}[/]  |  "
            f"LLM: [yellow]{llm_label}[/]  |  "
        )

        with ThreadPoolExecutor(max_workers=2) as pool:
            embed_future    = pool.submit(self._init_embed_model)
            llm_future      = pool.submit(self._init_llm)

            self.embed_model = embed_future.result()
            self.llm         = llm_future.result()
            LlamaSettings.llm = self.llm  

        logger.info("Initializing Node Parser...")
        self.node_parser = self._init_node_parser()
        
    def _ensure_directories(self):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        os.makedirs(config.LANCEDB_DIR, exist_ok=True) 

    def thai_tokenizer(text: str):
        return word_tokenize(text, engine="newmm")

    def _init_embed_model(self) -> HuggingFaceEmbedding:
        return HuggingFaceEmbedding(
            model_name=config.EMBED_MODEL_NAME,
            device = "cuda" if torch.cuda.is_available() else "cpu" , # เปลี่ยนเป็น "cpu" ได้ถ้าเครื่องไม่มีการ์ดจอ
            trust_remote_code=True
        )
    
    

    def _init_node_parser(self) -> SemanticSplitterNodeParser:
        return SemanticSplitterNodeParser.from_defaults(
            embed_model=self.embed_model,
            breakpoint_percentile_threshold=80,
            buffer_size=5
        )

    def _init_llm(self):
        if config.USE_GEMINI:
            return self._init_gemini_llm()
        if config.USE_OPENROUTER:
            return self._init_openrouter_llm()
        return self._init_ollama_llm()
    
    def _init_ollama_llm(self) -> Ollama:
        logger.info(f"Using [bold green]Ollama[/] — model: [yellow]{config.OLLAMA_MODEL}[/]")
        return Ollama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            request_timeout=600.00,
            context_window=config.CONTEXT_WINDOW,
            num_output=config.NUM_OUTPUT,
            system_prompt=config.SAFETY_SYSTEM_PROMPT,
        )
    
    def _init_openrouter_llm(self):

        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set in environment variables.")

        logger.info(
            f"Using [bold magenta]OpenRouter[/] — model: [yellow]{config.OPENROUTER_MODEL}[/]"
        )

        return OpenAILike(
            model=config.OPENROUTER_MODEL,
            api_base=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
            context_window=config.OPENROUTER_CONTEXT_WINDOW,
            max_tokens=4096,
            is_chat_model=True,
            system_prompt=config.SAFETY_SYSTEM_PROMPT,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost"),
                "X-Title": os.getenv("OPENROUTER_APP_TITLE", "RAG-App"),
            },
        )
    
    def _init_gemini_llm(self) -> Gemini:
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment variables.")

        logger.info(
            f"Using [bold blue]Google Gemini[/] — model: [yellow]{config.GEMINI_MODEL}[/]"
        )

        return Gemini(
            model=config.GEMINI_MODEL,
            api_key=config.GEMINI_API_KEY,
            max_tokens=config.NUM_OUTPUT,
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
        """
        โหลดหรือสร้าง Index ใหม่ โดยเพิ่มความสามารถในการสรุปเอกสาร (DocumentSummaryIndex)
        """
        try:
            existing_tables = self.lance_db.table_names()
            if config.TABLE_NAME in existing_tables:
                logger.info(f"Found existing LanceDB table: [green]{config.TABLE_NAME}[/]")
                
                # โหลด Vector Index เดิม
                index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                    embed_model=self.embed_model,
                )
                
                # โหลด nodes เข้า cache สำหรับ BM25
                self._load_nodes_cache_from_lancedb()
                return index
            else:
                # ถ้าไม่เจอ Table ให้ข้ามไปที่ Exception เพื่อสร้างใหม่
                raise ValueError(f"Table '{config.TABLE_NAME}' not found.")
                
        except Exception as e:
            logger.warning(f"Creating new index because: {e}")
            logger.info("Reading documents from DATA_DIR...")

            # 1. โหลดเอกสาร
            documents = []
            try:
                documents = SimpleDirectoryReader(config.DATA_DIR).load_data()
                # ใส่ metadata พื้นฐาน
                for d in documents:
                    d.metadata = d.metadata or {}
                    d.metadata.setdefault("channel_id", "global")
            except Exception as read_err:
                logger.error(f"Error reading directory: {read_err}")

            if not documents:
                logger.warning("No documents found. Returning an empty VectorStoreIndex.")
                return VectorStoreIndex([], storage_context=self.storage_context, embed_model=self.embed_model)

            # 3. สร้าง DocumentSummaryIndex (จุดที่ทำให้ RAG สรุปได้)
            # ตัวนี้จะวิ่งไปสั่ง LLM สรุปแต่ละไฟล์เก็บไว้เป็น Metadata พิเศษ
            logger.info("[bold yellow]Generating Document Summaries... (This may take a moment)[/]")
            
            # ดึง Nodes ทั้งหมดออกมา (ซึ่งตอนนี้จะมี Metadata สรุปติดมาด้วยแล้ว)
            nodes = self._build_summary_nodes(documents)

            # 4. เก็บลง cache สำหรับ BM25
            if nodes:
                self._add_nodes_to_cache(nodes)

            # 5. สร้าง VectorStoreIndex ตัวหลักที่จะใช้ใน Chat Engine
            logger.info("Finalizing VectorStoreIndex...")
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

            all_nodes = []
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
                
                all_nodes.append(node)
                
            self._add_nodes_to_cache(all_nodes)

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
                tokenizer=self.thai_tokenizer
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
        """
        เพิ่มเอกสารใหม่ โดยมีการทำ Summary ก่อนเก็บลง Vector Index
        """
        if not docs:
            logger.warning("No documents provided to add.")
            return

        logger.info(f"Adding [bold]{len(docs)}[/] documents with Auto-Summary...")

        # 1. เตรียม Metadata พื้นฐานสำหรับเอกสารใหม่
        for d in docs:
            d.metadata = d.metadata or {}
            # ตรวจสอบว่ามี channel_id หรือยัง ถ้าไม่มีให้ใส่ default
            if "channel_id" not in d.metadata:
                d.metadata["channel_id"] = "global"

        try:
     
            # 3. ดึง Nodes ที่ถูกสรุปและจัดการ Metadata เรียบร้อยแล้วออกมา
            new_nodes = self._build_summary_nodes(docs)  # แก้ให้ใช้เอกสารต้นฉบับที่มี metadata สรุปแล้ว

            # 4. Insert nodes เหล่านี้ลงใน VectorStoreIndex หลัก (LanceDB)
            if new_nodes:
                self.index.insert_nodes(new_nodes)
                self._add_nodes_to_cache(new_nodes)
                
                # 5. Clear chat_engines cache เพื่อให้การ Query ครั้งต่อไปเห็นข้อมูลใหม่
                with self.engine_lock:
                    self.chat_engines.clear()
                
                logger.info(f"[bold green]✅ Successfully added {len(new_nodes)} nodes with summaries to index.[/]")
            else:
                logger.warning("No nodes generated from the provided documents.")

        except Exception as e:
            logger.error(f"[bold red]❌ Failed to add documents:[/]\n{e}")
            raise e

    def delete_documents_by_metadata(self, metadata: Dict[str, Any]):
        where_clauses = [f"metadata.{k} = '{v}'" for k, v in metadata.items()]
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
    

    async def astream_query(
        self,
        question: str,
        channel_id: Union[str, int],
        sessions_id: Union[str, int, None] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream คำตอบแบบ token-by-token ผ่าน async generator"""
        logger.info(f"Streaming query: [bold cyan]{question}[/] (Channel: {channel_id})")

        chat_engine = self._get_chat_engine(str(channel_id), sessions_id)

        # CondensePlusContextChatEngine รองรับ astream_chat
        streaming_response = await chat_engine.astream_chat(question)

        async for token in streaming_response.async_response_gen():
            yield token
    
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
# 3. Global Instance (Lazy Loading)
# ==========================================

class LazyRAGEngine:
    def __init__(self):
        self._engine: Optional[RAGService] = None
        self._init_error: Optional[Exception] = None
        self._lock = RLock()

    def _get_engine(self) -> RAGService:
        if self._engine is not None:
            return self._engine

        with self._lock:
            if self._engine is not None:
                return self._engine

            if self._init_error is not None:
                raise RuntimeError("RAG Engine initialization previously failed") from self._init_error

            try:
                self._engine = RAGService()
            except Exception as e:
                self._init_error = e
                logger.exception("Failed to initialize RAG Engine")
                raise

        return self._engine

    def __getattr__(self, item):
        return getattr(self._get_engine(), item)

    def is_ready(self) -> bool:
        return self._engine is not None


rag_engine = LazyRAGEngine()