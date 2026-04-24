from email.mime import text
import asyncio
import os
import re
import logging
import urllib.request
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
from llama_index.core.callbacks.token_counting import get_tokens_from_response

from concurrent.futures import ThreadPoolExecutor

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from pythainlp.tokenize import word_tokenize
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.gemini import Gemini

from threading import RLock

import torch

# ✅ [STEP 2] Import SummaryWorker
from summary_worker import SummaryWorker, SummaryWorkerConfig

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
    
    # === Ollama Models ===
    EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://172.16.212.100:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "ministral-3:3b")

    # SummaryWorker fallback LLM (ใช้เฉพาะงาน summarize)
    SUMMARY_FALLBACK_OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "mistralai/ministral-3b-2512")
    SUMMARY_FALLBACK_OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    SUMMARY_FALLBACK_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    SUMMARY_FALLBACK_OPENROUTER_CONTEXT_WINDOW: int = int(os.getenv("OPENROUTER_CONTEXT_WINDOW", "262144"))
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Parameters
    CONTEXT_WINDOW: int = 32768
    NUM_OUTPUT: int = 512
    TOP_K: int = 5
    CHUNK_SIZE: int = 512
    
    # Prompts
    SAFETY_SYSTEM_PROMPT = (
        "คุณคือ AI ผู้ช่วยตอบคำถามจากเอกสาร\n\n"
        "กฎเหล็ก:\n"
        "1. ตอบโดยใช้ข้อมูลจาก [Context] ที่ให้มาเท่านั้น\n"
        "2. หากไม่พบคำตอบใน [Context] ให้ตอบว่า: "
        "'ขออภัย ฉันไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่มีอยู่ 😔 / "
        "Sorry, I could not find relevant information in the available documents. 😔'\n"
        "3. ห้ามใช้ความรู้ภายนอก ห้ามคาดเดา ห้ามสรุปเกินกว่าที่เขียนไว้\n\n"
        "ภาษา: ตอบด้วยภาษาเดียวกับที่ผู้ใช้ถาม (ไทยหรืออังกฤษ)"
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
# ==========================================

class OllamaTokenHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.latest_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @staticmethod
    def _to_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _build_usage(cls, prompt_tokens: Any, completion_tokens: Any, total_tokens: Any = None) -> Dict[str, int]:
        p = cls._to_int(prompt_tokens)
        c = cls._to_int(completion_tokens)
        t = cls._to_int(total_tokens)
        if t <= 0:
            t = p + c
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": t}

    def on_event_start(self, event_type, payload, event_id, **kwargs):
        pass  

    def on_event_end(self, event_type, payload, event_id, **kwargs):
        if event_type != CBEventType.LLM:
            return

        response = payload.get(EventPayload.RESPONSE)
        if response is None:
            return

        try:
            p, c = get_tokens_from_response(response)
            usage = self._build_usage(p, c)
            if usage["total_tokens"] > 0:
                self.latest_usage = usage
                return
        except Exception:
            pass

        raw = getattr(response, "raw", None)
        if not isinstance(raw, dict):
            return

        usage_data = raw.get("usage", raw.get("usage_metadata", {}))
        if isinstance(usage_data, dict):
            usage = self._build_usage(
                usage_data.get("prompt_tokens", usage_data.get("input_tokens", usage_data.get("prompt_token_count", 0))),
                usage_data.get("completion_tokens", usage_data.get("output_tokens", usage_data.get("candidates_token_count", 0))),
                usage_data.get("total_tokens", usage_data.get("total_token_count", 0)),
            )
            if usage["total_tokens"] > 0:
                self.latest_usage = usage
                return

        usage = self._build_usage(raw.get("prompt_eval_count", 0), raw.get("eval_count", 0))
        if usage["total_tokens"] > 0:
            self.latest_usage = usage
    
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

        logger.info("Loading models in parallel...")
        self._init_models_parallel_sync()

        logger.info("Connecting to [bright_blue]LanceDB...[/]")
        self.lance_db, self.vector_store = self._init_lancedb()
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        logger.info("Loading/Creating Vector Index...")
        self.index = self._load_or_create_index()

        logger.info(f"Initializing Redis Chat Store at [italic hot_pink]{config.REDIS_URL}[/]...")
        try:
            self.chat_store = RedisChatStore(redis_url=config.REDIS_URL,ttl=86400)
            if self.chat_store._redis_client.ping():
                logger.info("[bold green]✅ Connected to Redis Chat Store successfully.[/]")
            else:
                logger.warning("[bold yellow]⚠️ Failed to connect Redis Chat Store.[/]")
        except Exception as e:
            logger.error(f"[bold red]❌ Failed to connect to Redis:[/]\n{e}")
            raise e

        summary_fallback_llm = self._init_summary_fallback_llm()

        # ✅ [STEP 2 — จุดที่ 1] Init และ start SummaryWorker หลัง Redis พร้อม
        self.summary_worker = SummaryWorker(
            llm=self.llm,
            chat_store=self.chat_store,
            nodes_lock=self.nodes_lock,
            nodes_cache=self.nodes_cache,
            fallback_llm=summary_fallback_llm,
            cfg=SummaryWorkerConfig(
                redis_ttl_seconds=None,
                max_retries=3,
                retry_delay_seconds=5.0,
                max_queue_size=500,
            ),
        )

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

            if len(self.nodes_cache) > config.MAX_CACHE_NODES:
                overflow = len(self.nodes_cache) - config.MAX_CACHE_NODES
                keys_to_remove = list(self.nodes_cache.keys())[:overflow]
                for k in keys_to_remove:
                    del self.nodes_cache[k]

        logger.info(f"nodes_cache size: {len(self.nodes_cache)} (+{len(nodes)} added)")
    
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
            device="cuda" if torch.cuda.is_available() else "cpu",
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
        self._log_ollama_connectivity()
        return Ollama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            request_timeout=600.00,
            context_window=config.CONTEXT_WINDOW,
            num_output=config.NUM_OUTPUT,
            system_prompt=config.SAFETY_SYSTEM_PROMPT,
        )

    def _init_summary_fallback_llm(self):
        model_name = config.SUMMARY_FALLBACK_OPENROUTER_MODEL.strip()
        if not model_name:
            logger.info(
                "Summary fallback LLM is disabled (SUMMARY_FALLBACK_OPENROUTER_MODEL is empty)"
            )
            return None

        api_key = config.SUMMARY_FALLBACK_OPENROUTER_API_KEY.strip()
        if not api_key:
            logger.warning(
                "[bold yellow]⚠️ Summary fallback LLM disabled: SUMMARY_FALLBACK_OPENROUTER_API_KEY is empty[/]"
            )
            return None

        base_url = config.SUMMARY_FALLBACK_OPENROUTER_BASE_URL
        logger.info(
            f"Configuring SummaryWorker fallback LLM — OpenRouter model: [yellow]{model_name}[/]"
        )
        try:
            return OpenAILike(
                model=model_name,
                api_base=base_url,
                api_key=api_key,
                context_window=config.SUMMARY_FALLBACK_OPENROUTER_CONTEXT_WINDOW,
                max_tokens=4096,
                is_chat_model=True,
                system_prompt=config.SAFETY_SYSTEM_PROMPT,
                default_headers={
                    "HTTP-Referer": os.getenv(
                        "SUMMARY_FALLBACK_OPENROUTER_REFERER",
                        os.getenv("OPENROUTER_REFERER", "http://localhost"),
                    ),
                    "X-Title": os.getenv(
                        "SUMMARY_FALLBACK_OPENROUTER_APP_TITLE",
                        os.getenv("OPENROUTER_APP_TITLE", "RAG-App"),
                    ),
                },
            )
        except Exception as e:
            logger.warning(
                f"[bold yellow]⚠️ Failed to initialize SummaryWorker fallback LLM:[/] {e}"
            )
            return None

    def _log_ollama_connectivity(self, base_url: str | None = None) -> None:
        resolved_base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        health_url = f"{resolved_base_url}/api/tags"
        try:
            request = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                if 200 <= response.status < 300:
                    logger.info(
                        f"[bold green]✅ Ollama is reachable at[/] [yellow]{resolved_base_url}[/]"
                    )
                    return

                logger.warning(
                    f"[bold yellow]⚠️ Ollama responded with unexpected status {response.status} at[/] "
                    f"[yellow]{resolved_base_url}[/]"
                )
        except Exception as e:
            logger.warning(
                f"[bold yellow]⚠️ Cannot connect to Ollama at[/] [yellow]{resolved_base_url}[/]: {e}"
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
                
                self._load_nodes_cache_from_lancedb()
                return index
            else:
                raise ValueError(f"Table '{config.TABLE_NAME}' not found.")
                
        except Exception as e:
            logger.warning(f"Creating new index because: {e}")
            logger.info("Reading documents from DATA_DIR...")

            documents = []
            try:
                documents = SimpleDirectoryReader(config.DATA_DIR).load_data()
                for d in documents:
                    d.metadata = d.metadata or {}
                    d.metadata.setdefault("channel_id", "global")
            except Exception as read_err:
                logger.error(f"Error reading directory: {read_err}")

            if not documents:
                logger.warning("No documents found. Returning an empty VectorStoreIndex.")
                return VectorStoreIndex([], storage_context=self.storage_context, embed_model=self.embed_model)

            logger.info("[bold yellow]Parsing documents into nodes...[/]")
            nodes = self.node_parser.get_nodes_from_documents(documents)

            if nodes:
                self._add_nodes_to_cache(nodes)

            logger.info("Finalizing VectorStoreIndex...")
            return VectorStoreIndex(
                nodes,
                storage_context=self.storage_context,
                embed_model=self.embed_model
            )
    
    def _load_nodes_cache_from_lancedb(self):
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

    def _strip_think(self, text: str) -> str:
        if not text:
            return text
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"(?i)(^|\n)\s*(analysis:|reasoning:|thoughts?:).*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
        return text.strip()

    @staticmethod
    def _build_usage_dict(prompt_tokens: Any, completion_tokens: Any, total_tokens: Any = None) -> Dict[str, int]:
        try:
            p = int(prompt_tokens or 0)
        except (TypeError, ValueError):
            p = 0
        try:
            c = int(completion_tokens or 0)
        except (TypeError, ValueError):
            c = 0
        try:
            t = int(total_tokens or 0)
        except (TypeError, ValueError):
            t = 0

        if t <= 0:
            t = p + c

        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": t}

    def _resolve_usage(self, response: Any) -> Dict[str, int]:
        usage = dict(self.token_handler.latest_usage)
        if usage.get("total_tokens", 0) > 0:
            return usage

        if response is None:
            return usage

        try:
            p, c = get_tokens_from_response(response)
            usage = self._build_usage_dict(p, c)
            if usage["total_tokens"] > 0:
                return usage
        except Exception:
            pass

        raw = getattr(response, "raw", None)
        if isinstance(raw, dict):
            usage_data = raw.get("usage", raw.get("usage_metadata", {}))
            if isinstance(usage_data, dict):
                usage = self._build_usage_dict(
                    usage_data.get("prompt_tokens", usage_data.get("input_tokens", usage_data.get("prompt_token_count", 0))),
                    usage_data.get("completion_tokens", usage_data.get("output_tokens", usage_data.get("candidates_token_count", 0))),
                    usage_data.get("total_tokens", usage_data.get("total_token_count", 0)),
                )
                if usage["total_tokens"] > 0:
                    return usage

            usage = self._build_usage_dict(raw.get("prompt_eval_count", 0), raw.get("eval_count", 0))
            if usage["total_tokens"] > 0:
                return usage

        return usage

    def _extract_source_filenames(self, source_nodes: Optional[List[Any]]) -> List[str]:
        if not source_nodes:
            return []

        file_names = set()
        for source_node in source_nodes:
            node = getattr(source_node, "node", source_node)
            metadata = getattr(node, "metadata", {}) or {}
            file_name = (
                metadata.get("filename")
                or metadata.get("file_name")
                or metadata.get("source")
            )
            if file_name:
                file_names.add(str(file_name))

        return sorted(file_names)

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

        vector_retriever = self.index.as_retriever(
            similarity_top_k=config.TOP_K * 2,
            filters=filters,
        )

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
            retrievers = [vector_retriever]
            logger.info("No nodes in cache yet, using vector-only retriever")

        hybrid_retriever = QueryFusionRetriever(
            retrievers=retrievers,
            similarity_top_k=config.TOP_K,
            num_queries=1,
            mode="reciprocal_rerank",
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
        เพิ่มเอกสารใหม่และแยกเป็น nodes ก่อนเก็บลง Vector Index
        """
        if not docs:
            logger.warning("No documents provided to add.")
            return

        logger.info(f"Adding [bold]{len(docs)}[/] documents...")

        for d in docs:
            d.metadata = d.metadata or {}
            if "channel_id" not in d.metadata:
                d.metadata["channel_id"] = "global"

        try:
            new_nodes = self.node_parser.get_nodes_from_documents(docs)

            if new_nodes:
                self.index.insert_nodes(new_nodes)
                self._add_nodes_to_cache(new_nodes)
                
                with self.engine_lock:
                    self.chat_engines.clear()
                
                logger.info(f"[bold green]✅ Successfully added {len(new_nodes)} nodes to index.[/]")

                # ✅ [STEP 2 — จุดที่ 2] Enqueue summary job ต่อ 1 ไฟล์ หลัง insert เสร็จ
                # รวบรวม (channel_id, file_name) ที่ไม่ซ้ำจากเอกสารที่เพิ่งเพิ่ม
                files_added: set[tuple[str, str]] = {
                    (
                        str(d.metadata.get("channel_id", "global")),
                        str(d.metadata.get("file_name", d.metadata.get("filename", "unknown"))),
                    )
                    for d in docs
                }
                for channel_id, file_name in files_added:
                    self.summary_worker.enqueue(channel_id, file_name)
                    logger.info(f"📥 Summary enqueued for: {file_name} (channel={channel_id})")

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

                # ✅ [STEP 2 — จุดที่ 3] เก็บ (channel_id, file_name) ก่อนลบ nodes
                # เพื่อนำไปลบ summary จาก Redis ด้านล่าง
                deleted_files: set[tuple[str, str]] = {
                    (
                        str(self.nodes_cache[node_id].metadata.get("channel_id", "global")),
                        str(self.nodes_cache[node_id].metadata.get("file_name", "unknown")),
                    )
                    for node_id in keys_to_delete
                }

                for k in keys_to_delete:
                    del self.nodes_cache[k]

                removed = before - len(self.nodes_cache)

            logger.info(f"[bold green]✅ Delete successful. Removed {removed} nodes from cache.[/]")

            # ✅ [STEP 2 — จุดที่ 3 ต่อ] ลบ summary ออกจาก Redis แบบ fire-and-forget
            import asyncio
            for channel_id, file_name in deleted_files:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # เรียกจาก async context (FastAPI) → create_task ได้เลย
                        asyncio.create_task(
                            self.summary_worker.delete(channel_id, file_name)
                        )
                    else:
                        # เรียกจาก sync context → run จนจบ
                        loop.run_until_complete(
                            self.summary_worker.delete(channel_id, file_name)
                        )
                except Exception as del_err:
                    logger.warning(f"Could not delete summary for {file_name}: {del_err}")

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

    # ✅ [STEP 3] Pattern ตรวจ overview question — ปรับ keyword เพิ่มได้ตามต้องการ
    _OVERVIEW_PATTERN = re.compile(
        r"(สรุป|เกี่ยวกับอะไร|พูดถึงอะไร|ภาพรวม|มีอะไรบ้าง|overview|summary|เนื้อหาหลัก|ประเด็นหลัก|บอกเกี่ยวกับ|เอกสารนี้คืออะไร)",
        re.IGNORECASE,
    )

    def _is_overview_question(self, question: str) -> bool:
        return bool(self._OVERVIEW_PATTERN.search(question))

    async def _build_summary_response(
        self,
        question: str,
        channel_id: str,
    ) -> Dict[str, Any] | None:
        """
        ดึง summary ทุกไฟล์ของ channel นี้จาก Redis แล้วให้ LLM ตอบ
        คืน None ถ้าไม่มี summary พร้อมเลย (fallback ไป RAG ปกติ)
        """
        summaries: dict[str, str] = await self.summary_worker.get_all_for_channel(channel_id)

        if not summaries:
            logger.info("[yellow]No summaries ready yet, falling back to RAG[/]")
            return None

        # จัดรูป context: [ชื่อไฟล์]\nสรุป...
        context_parts = [
            f"[{file_name}]\n{summary_text}"
            for file_name, summary_text in summaries.items()
        ]
        context = "\n\n---\n\n".join(context_parts)

        prompt = (
            f"จากสรุปเอกสารด้านล่าง จงตอบคำถามต่อไปนี้อย่างละเอียดเป็นภาษาไทย\n"
            f"คำถาม: {question}\n\n"
            f"[สรุปเอกสาร]\n{context}"
        )

        logger.info(f"📋 Overview route — using {len(summaries)} summary(s) from Redis")
        raw = await self.llm.acomplete(prompt)
        answer = self._strip_think(str(raw))
        usage = self._resolve_usage(raw)

        return {
            "answer": answer,
            "usage": usage,
            "sources": sorted(summaries.keys()),
            "from_summary": True,   # ← ให้ API layer รู้ว่ามาจาก summary cache
        }

    async def _build_summary_progress_status(self, channel_id: str) -> Dict[str, Any]:
        with self.nodes_lock:
            file_names = sorted(
                {
                    str(node.metadata.get("file_name") or node.metadata.get("filename") or "")
                    for node in self.nodes_cache.values()
                    if str(node.metadata.get("channel_id", "")) == channel_id
                }
            )

        file_names = [name for name in file_names if name]

        if not file_names:
            return {
                "ready": False,
                "status": "not_found",
                "ready_files": 0,
                "total_files": 0,
                "processing_files": 0,
                "queued_files": 0,
                "failed_files": 0,
                "message": "Summary ภาพรวมยังไม่พร้อม และยังไม่พบไฟล์ใน channel นี้",
            }

        statuses = await asyncio.gather(
            *[
                self.summary_worker.is_ready(channel_id=channel_id, file_name=file_name)
                for file_name in file_names
            ],
            return_exceptions=True,
        )

        ready_files = 0
        processing_files = 0
        queued_files = 0
        failed_files = 0
        not_queued_files = 0

        for status in statuses:
            if isinstance(status, Exception):
                continue

            if status.get("ready"):
                ready_files += 1

            state = status.get("status")
            if state == "processing":
                processing_files += 1
            elif state == "queued":
                queued_files += 1
            elif state == "failed":
                failed_files += 1
            elif state == "not_queued":
                not_queued_files += 1

        activities: list[str] = []
        if processing_files > 0:
            activities.append(f"กำลังสรุป {processing_files} ไฟล์")
        if queued_files > 0:
            activities.append(f"รอคิว {queued_files} ไฟล์")
        if not_queued_files > 0:
            activities.append(f"เตรียมส่งคิว {not_queued_files} ไฟล์")
        if failed_files > 0:
            activities.append(f"สรุปไม่สำเร็จ {failed_files} ไฟล์")

        activity_text = ", ".join(activities) if activities else "กำลังเตรียมข้อมูลสรุป"
        total_files = len(file_names)
        message = (
            f"Summary ภาพรวมยังไม่พร้อม ({ready_files}/{total_files} ไฟล์พร้อมใช้งาน) "
            f"ตอนนี้ระบบ{activity_text} จึงใช้โหมดค้นเอกสารปกติชั่วคราว"
        )

        return {
            "ready": False,
            "status": "pending",
            "ready_files": ready_files,
            "total_files": total_files,
            "processing_files": processing_files,
            "queued_files": queued_files,
            "failed_files": failed_files,
            "message": message,
        }

    @staticmethod
    def _prepend_notice(answer: str, notice: Optional[str]) -> str:
        if not notice:
            return answer
        if not answer:
            return notice
        return f"{notice}\n\n{answer}"

    async def aquery(self, question: str, channel_id: Union[str, int], sessions_id: Union[str, int, None] = None) -> Dict[str, Any]:
        logger.info(f"Querying: [bold cyan]{question}[/] (Channel: {channel_id})")
        self.token_handler.latest_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        summary_status: Dict[str, Any] | None = None

        # ✅ [STEP 3] Overview routing — ถ้าเป็นคำถาม overview ให้ใช้ summary จาก Redis ก่อน
        if self._is_overview_question(question):
            logger.info("[bold magenta]ROUTE (query): SUMMARY[/] — overview question detected")
            summary_result = await self._build_summary_response(question, str(channel_id))
            if summary_result:
                logger.info("[bold magenta]ROUTE SELECTED (query): SUMMARY[/] — answered from summary cache")
                return summary_result
            # summary ยังไม่พร้อม → fall through ไป RAG ปกติด้านล่าง
            summary_status = await self._build_summary_progress_status(str(channel_id))
            logger.info(f"[bold yellow]SUMMARY STATUS (query):[/] {summary_status['message']}")
            logger.info("[bold yellow]ROUTE FALLBACK (query): RAG HYBRID[/] — summary not ready")
        else:
            logger.info("[bold cyan]ROUTE SELECTED (query): RAG HYBRID[/] — non-overview question")

        logger.info("[bold cyan]Using normal RAG retrieval pipeline (hybrid retrieval)[/]")
        chat_engine = self._get_chat_engine(str(channel_id), sessions_id)
        response = await chat_engine.achat(question)

        answer_text = self._strip_think(str(response))
        if summary_status:
            answer_text = self._prepend_notice(answer_text, summary_status.get("message"))
        token_usage = self._resolve_usage(response)

        if not response.source_nodes:
            logger.info("[yellow]No source nodes found.[/]")
            result = {
                "answer": self._prepend_notice(
                    "ตอนนี้ยังไม่มีเอกสารที่ใช้ตอบได้เลยนะ🤔 รบกวนเพิ่มเอกสารก่อนนะคะ😊",
                    summary_status.get("message") if summary_status else None,
                ),
                "usage": token_usage,
                "sources": [],
            }
            if summary_status:
                result["summary_status"] = summary_status
                result["from_summary"] = False
            return result
            
        TOP_SCORE_THRESHOLD = 0.35
        top_score = response.source_nodes[0].score or 0
        if top_score < TOP_SCORE_THRESHOLD:
            result = {
                "answer": self._prepend_notice(
                    "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร 😔",
                    summary_status.get("message") if summary_status else None,
                ),
                "usage": token_usage,
                "sources": [],
            }
            if summary_status:
                result["summary_status"] = summary_status
                result["from_summary"] = False
            return result
        
        file_names = self._extract_source_filenames(response.source_nodes)
        if file_names:
            logger.info(f"Sources used: [green]{file_names}[/]")

        result = {
            "answer": answer_text,
            "usage": token_usage,
            "sources": file_names,
        }
        if summary_status:
            result["summary_status"] = summary_status
            result["from_summary"] = False
        return result

    async def astream_query(
        self,
        question: str,
        channel_id: Union[str, int],
        sessions_id: Union[str, int, None] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        logger.info(f"Streaming query: [bold cyan]{question}[/] (Channel: {channel_id})")
        self.token_handler.latest_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        summary_status: Dict[str, Any] | None = None
        full_answer = ""

        # ✅ [STEP 3] Overview routing สำหรับ streaming
        if self._is_overview_question(question):
            logger.info("[bold magenta]ROUTE (stream): SUMMARY[/] — overview question detected")
            summary_result = await self._build_summary_response(question, str(channel_id))
            if summary_result:
                logger.info("[bold magenta]ROUTE SELECTED (stream): SUMMARY[/] — answered from summary cache")
                # จำลอง stream ทีละ token เพื่อให้ client ไม่ต้องแยก code path
                for token in summary_result["answer"]:
                    yield {"type": "token", "token": token}
                yield {
                    "type": "meta",
                    "answer": summary_result["answer"],
                    "sources": summary_result["sources"],
                    "usage": summary_result["usage"],
                    "from_summary": True,
                }
                return
            # summary ยังไม่พร้อม → fall through ไป RAG ปกติด้านล่าง
            summary_status = await self._build_summary_progress_status(str(channel_id))
            logger.info(f"[bold yellow]SUMMARY STATUS (stream):[/] {summary_status['message']}")
            logger.info("[bold yellow]ROUTE FALLBACK (stream): RAG HYBRID[/] — summary not ready")
        else:
            logger.info("[bold cyan]ROUTE SELECTED (stream): RAG HYBRID[/] — non-overview question")

        if summary_status:
            notice = f"{summary_status['message']}\n\n"
            for token in notice:
                full_answer += token
                yield {"type": "token", "token": token}

        logger.info("[bold cyan]Using normal RAG retrieval pipeline (hybrid retrieval) for streaming[/]")
        chat_engine = self._get_chat_engine(str(channel_id), sessions_id)

        streaming_response = await chat_engine.astream_chat(question)

        async for token in streaming_response.async_response_gen():
            full_answer += token
            yield {"type": "token", "token": token}

        sources = self._extract_source_filenames(streaming_response.source_nodes)
        usage = self._resolve_usage(streaming_response)

        meta_payload: Dict[str, Any] = {
            "type": "meta",
            "answer": self._strip_think(full_answer),
            "sources": sources,
            "usage": usage,
        }
        if summary_status:
            meta_payload["summary_status"] = summary_status
            meta_payload["from_summary"] = False

        yield meta_payload
    
    def debug_list_docs_by_channel(self, channel_id: int):
        try:
            table = self.lance_db.open_table(config.TABLE_NAME)
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
