"""
summary_worker.py
-----------------
Background summarization worker สำหรับ RAG Engine
- รัน summarize ใน asyncio background task ไม่บล็อก upload
- เก็บผลลัพธ์ลง Redis ด้วย key  `summary:{channel_id}:{file_name}`
- รองรับ retry, graceful shutdown, และ dedup job
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING

from llama_index.core.indices import SummaryIndex

if TYPE_CHECKING:
    from llama_index.storage.chat_store.redis import RedisChatStore
    from llama_index.core.schema import TextNode

logger = logging.getLogger("RAG_ENGINE")

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

@dataclass
class SummaryWorkerConfig:
    redis_ttl_seconds: int = 60 * 60 * 24 * 7   # 7 วัน
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    summarize_prompt: str = (
        "สรุปเนื้อหาสำคัญของเอกสารนี้เป็นภาษาไทย "
        "ให้ครอบคลุมประเด็นหลักและรายละเอียดที่สำคัญ"
    )
    # ป้องกัน queue ใหญ่เกินไปถ้า upload พร้อมกันเยอะ
    max_queue_size: int = 500


# ──────────────────────────────────────────────
# Job
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class SummaryJob:
    channel_id: str
    file_name: str

    @property
    def redis_key(self) -> str:
        return f"summary:{self.channel_id}:{self.file_name}"


# ──────────────────────────────────────────────
# Worker
# ──────────────────────────────────────────────

class SummaryWorker:
    """
    Async background worker ที่รัน summarize และเก็บผลลัพธ์ลง Redis

    Usage (ใน RAGService.__init__):
        self.summary_worker = SummaryWorker(llm, chat_store, nodes_lock, nodes_cache)
        self.summary_worker.start()

    Usage (ใน add_documents):
        self.summary_worker.enqueue(channel_id, file_name)

    Usage (ใน aquery):
        summary = await self.summary_worker.get(channel_id, file_name)
    """

    def __init__(
        self,
        llm,
        chat_store: "RedisChatStore",
        nodes_lock: Lock,
        nodes_cache: dict[str, "TextNode"],
        cfg: SummaryWorkerConfig | None = None,
    ):
        self.llm = llm
        self.chat_store = chat_store
        self.nodes_lock = nodes_lock
        self.nodes_cache = nodes_cache
        self.cfg = cfg or SummaryWorkerConfig()

        self._queue: asyncio.Queue[SummaryJob] = asyncio.Queue(
            maxsize=self.cfg.max_queue_size
        )
        # เก็บ jobs ที่อยู่ใน queue / กำลัง process → ป้องกัน enqueue ซ้ำ
        self._pending: set[SummaryJob] = set()
        self._pending_lock = asyncio.Lock()

        self._task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

    # ── Lifecycle ──────────────────────────────

    def start(self) -> None:
        """เรียกครั้งเดียวหลัง event loop เริ่มทำงาน (ใน RAGService.__init__)"""
        if self._task and not self._task.done():
            logger.warning("SummaryWorker already running, skipping start()")
            return
        self._task = asyncio.create_task(
            self._worker_loop(), name="summary_worker"
        )
        logger.info("[bold cyan]📝 SummaryWorker started[/]")

    async def shutdown(self) -> None:
        """
        Graceful shutdown — รอ queue หมดก่อน แล้วค่อย cancel task
        เรียกใน app lifespan / shutdown event
        """
        logger.info("SummaryWorker shutting down — draining queue...")
        await self._queue.join()          # รอให้ job ที่ค้างอยู่เสร็จก่อน
        self._shutdown_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[bold green]✅ SummaryWorker shut down cleanly[/]")

    # ── Public API ────────────────────────────

    def enqueue(self, channel_id: str, file_name: str) -> bool:
        """
        ใส่งานเข้า queue แบบ non-blocking
        คืน True ถ้าใส่สำเร็จ, False ถ้า job นั้นอยู่ใน queue แล้ว
        """
        job = SummaryJob(channel_id=str(channel_id), file_name=str(file_name))

        # dedup — ถ้า job นี้ยังค้างอยู่ใน queue ไม่ต้องใส่ซ้ำ
        # ใช้ asyncio.Lock ไม่ได้ตรงนี้เพราะอาจเรียกจาก sync context
        # → ใช้ set ธรรมดา accept race condition เล็กน้อย (worst case: summarize ซ้ำ 1 ครั้ง)
        if job in self._pending:
            logger.debug(f"Summary job already pending, skipped: {job.redis_key}")
            return False

        try:
            self._queue.put_nowait(job)
            self._pending.add(job)
            logger.info(
                f"📥 Enqueued summary job — channel={channel_id} file={file_name} "
                f"(queue size: {self._queue.qsize()})"
            )
            return True
        except asyncio.QueueFull:
            logger.warning(
                f"[yellow]⚠️ Summary queue full ({self.cfg.max_queue_size}), "
                f"dropping job: {job.redis_key}[/]"
            )
            return False

    async def get(self, channel_id: str, file_name: str) -> str | None:
        """
        ดึง summary จาก Redis
        คืน None ถ้ายังไม่เสร็จ (worker ยังประมวลผลอยู่)
        """
        job = SummaryJob(channel_id=str(channel_id), file_name=str(file_name))
        return await self._redis_get(job.redis_key)

    async def get_all_for_channel(self, channel_id: str) -> dict[str, str]:
        """
        ดึง summary ทุกไฟล์ของ channel นี้จาก Redis
        คืน dict[file_name, summary_text]
        """
        pattern = f"summary:{channel_id}:*"
        try:
            keys: list[bytes] = await asyncio.to_thread(
                self.chat_store._redis_client.keys, pattern
            )
            result: dict[str, str] = {}
            for key_bytes in keys:
                key = key_bytes.decode()
                raw = await self._redis_get(key)
                if raw:
                    # key format: summary:{channel_id}:{file_name}
                    file_name = key.split(":", 2)[-1]
                    result[file_name] = raw
            return result
        except Exception as e:
            logger.warning(f"get_all_for_channel failed: {e}")
            return {}

    async def delete(self, channel_id: str, file_name: str) -> None:
        """ลบ summary ออกจาก Redis (เรียกตอน delete document)"""
        job = SummaryJob(channel_id=str(channel_id), file_name=str(file_name))
        try:
            await asyncio.to_thread(
                self.chat_store._redis_client.delete, job.redis_key
            )
            logger.info(f"🗑️ Deleted summary: {job.redis_key}")
        except Exception as e:
            logger.warning(f"Failed to delete summary {job.redis_key}: {e}")

    async def is_ready(self, channel_id: str, file_name: str) -> bool:
        """ตรวจว่า summary พร้อมใช้แล้วหรือยัง"""
        summary = await self.get(channel_id, file_name)
        return summary is not None

    # ── Internal — Worker Loop ─────────────────

    async def _worker_loop(self) -> None:
        logger.info("SummaryWorker loop started")
        while not self._shutdown_event.is_set():
            try:
                # timeout เพื่อให้ loop ตรวจ shutdown_event ได้
                job: SummaryJob = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._process_with_retry(job)
            finally:
                self._queue.task_done()
                self._pending.discard(job)

        logger.info("SummaryWorker loop exited")

    async def _process_with_retry(self, job: SummaryJob) -> None:
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                await self._summarize_and_store(job)
                return  # สำเร็จ ออกจาก loop ได้เลย
            except Exception as e:
                if attempt < self.cfg.max_retries:
                    logger.warning(
                        f"[yellow]Summary attempt {attempt}/{self.cfg.max_retries} "
                        f"failed for {job.file_name}: {e} — retrying in "
                        f"{self.cfg.retry_delay_seconds}s[/]"
                    )
                    await asyncio.sleep(self.cfg.retry_delay_seconds)
                else:
                    logger.error(
                        f"[bold red]❌ Summary failed after {self.cfg.max_retries} "
                        f"attempts for {job.file_name}: {e}[/]"
                    )

    async def _summarize_and_store(self, job: SummaryJob) -> None:
        logger.info(
            f"🔄 Summarizing: [bold]{job.file_name}[/] (channel={job.channel_id})"
        )

        # 1. ดึง nodes ของ file นี้จาก nodes_cache
        nodes = self._get_nodes_for_job(job)

        if not nodes:
            logger.warning(
                f"[yellow]No nodes found for {job.file_name} "
                f"in channel {job.channel_id}, skipping[/]"
            )
            return

        logger.info(f"Found {len(nodes)} nodes for {job.file_name}")

        # 2. Summarize ด้วย SummaryIndex
        summary_index = SummaryIndex(nodes)
        engine = summary_index.as_query_engine(
            llm=self.llm,
            response_mode="tree_summarize",
            use_async=True,
        )
        result = await engine.aquery(self.cfg.summarize_prompt)
        summary_text = str(result).strip()

        if not summary_text:
            raise ValueError("Summarization returned empty result")

        # 3. เก็บลง Redis
        await asyncio.to_thread(
            self.chat_store._redis_client.setex,
            job.redis_key,
            self.cfg.redis_ttl_seconds,
            summary_text,
        )

        logger.info(
            f"[bold green]✅ Summary stored[/] — key={job.redis_key} "
            f"({len(summary_text)} chars)"
        )

    def _get_nodes_for_job(self, job: SummaryJob) -> list:
        """ดึง nodes ที่ตรงกับ file_name + channel_id จาก nodes_cache"""
        with self.nodes_lock:
            return [
                node
                for node in self.nodes_cache.values()
                if (
                    node.metadata.get("file_name") == job.file_name
                    and str(node.metadata.get("channel_id", "")) == job.channel_id
                )
            ]

    # ── Internal — Redis Helpers ───────────────

    async def _redis_get(self, key: str) -> str | None:
        try:
            raw: bytes | None = await asyncio.to_thread(
                self.chat_store._redis_client.get, key
            )
            return raw.decode() if raw else None
        except Exception as e:
            logger.warning(f"Redis GET failed (key={key}): {e}")
            return None

    # ── Diagnostics ───────────────────────────

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def pending_jobs(self) -> list[SummaryJob]:
        return list(self._pending)

    def __repr__(self) -> str:
        return (
            f"<SummaryWorker queue_size={self.queue_size} "
            f"pending={len(self._pending)}>"
        )