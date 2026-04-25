"""
summary_worker.py
-----------------
Background summarization worker สำหรับ RAG Engine
- รัน summarize ใน asyncio background task ไม่บล็อก upload
- เก็บผลลัพธ์ลง Redis ด้วย key  `summary:{channel_id}:{file_name}`
- เก็บ quick questions ด้วย key `quick_questions:{channel_id}:{file_name}`
- รองรับ retry, graceful shutdown, และ dedup job
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any

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
    redis_ttl_seconds: int | None = None
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    summarize_prompt: str = (
        "สรุปเนื้อหาสำคัญของเอกสารนี้เป็นภาษาไทย "
        "ให้ครอบคลุมประเด็นหลักและรายละเอียดที่สำคัญ"
    )
    # ป้องกัน queue ใหญ่เกินไปถ้า upload พร้อมกันเยอะ
    max_queue_size: int = 500
    quick_questions_enabled: bool = True
    quick_questions_count: int = 5
    quick_questions_max_summary_chars: int = 6000
    quick_questions_prompt: str = (
        "จาก summary ด้านล่าง ให้สร้างคำถามปูพื้นภาษาไทย จำนวน {count} ข้อ\n"
        "เป้าหมาย: ช่วยผู้ใช้ที่ยังไม่รู้จักเอกสารนี้เลยให้เริ่มต้นสนทนาได้\n\n"
        "รูปแบบที่ต้องการ:\n"
        "- 'X คืออะไร?'\n"
        "- 'ใครคือ Y?'\n"
        "- 'Z มีไว้เพื่ออะไร?'\n\n"
        "กติกา:\n"
        "- X/Y/Z ต้องเป็นชื่อ entity จริงจากเอกสารนี้ ห้ามสร้างขึ้นมาเอง\n"
        "- ไม่เกิน 10 คำต่อข้อ\n"
        "- ครอบคลุมหลาย entity ไม่ถามซ้ำกัน\n"
        "Output: JSON array ของ string เท่านั้น ห้ามมีข้อความอื่น ห้ามใส่เลขหรือ bullet\n\n"
        "ชื่อไฟล์: {file_name}\n"
        "[SUMMARY]\n{summary}\n"
    )


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

    @property
    def quick_questions_redis_key(self) -> str:
        return f"quick_questions:{self.channel_id}:{self.file_name}"


@dataclass(frozen=True)
class SummaryFailureInfo:
    attempts: int
    error: str


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
        fallback_llm: Any | None = None,
        cfg: SummaryWorkerConfig | None = None,
    ):
        self.llm = llm
        self.fallback_llm = fallback_llm
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

        # state สำหรับรายงานความพร้อมของ summary ให้ผู้ใช้
        self._active_job: SummaryJob | None = None
        self._attempts: dict[SummaryJob, int] = {}
        self._failed_jobs: dict[SummaryJob, SummaryFailureInfo] = {}

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
            self._failed_jobs.pop(job, None)
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

    async def get_quick_questions(self, channel_id: str, file_name: str) -> list[str] | None:
        """ดึง quick questions ของไฟล์เดียวจาก Redis"""
        job = SummaryJob(channel_id=str(channel_id), file_name=str(file_name))
        payload = await self._redis_get(job.quick_questions_redis_key)
        if not payload:
            return None

        try:
            parsed = json.loads(payload)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception as e:
            logger.warning(
                f"Failed to parse quick questions payload for {job.quick_questions_redis_key}: {e}"
            )
        return None

    async def get_all_quick_questions_for_channel(self, channel_id: str) -> dict[str, list[str]]:
        """
        ดึง quick questions ทุกไฟล์ของ channel นี้จาก Redis
        คืน dict[file_name, quick_questions]
        """
        pattern = f"quick_questions:{channel_id}:*"
        try:
            keys: list[bytes] = await asyncio.to_thread(
                self.chat_store._redis_client.keys, pattern
            )
            result: dict[str, list[str]] = {}
            for key_bytes in keys:
                key = key_bytes.decode()
                raw = await self._redis_get(key)
                if not raw:
                    continue

                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = None

                if isinstance(parsed, list):
                    questions = [str(item).strip() for item in parsed if str(item).strip()]
                    if questions:
                        file_name = key.split(":", 2)[-1]
                        result[file_name] = questions
            return result
        except Exception as e:
            logger.warning(f"get_all_quick_questions_for_channel failed: {e}")
            return {}

    async def regenerate_quick_questions_for_channel(self, channel_id: str) -> dict[str, Any]:
        """
        ลบ quick questions เดิมของ channel แล้วสร้างใหม่จาก summary ที่มีใน Redis
        """
        real_channel_id = str(channel_id)
        deleted_quick_question_keys = await self._redis_delete_keys_by_pattern(
            f"quick_questions:{real_channel_id}:*"
        )

        summaries = await self.get_all_for_channel(real_channel_id)
        if not summaries:
            return {
                "channel_id": real_channel_id,
                "deleted_quick_question_keys": deleted_quick_question_keys,
                "total_summaries": 0,
                "regenerated_files": 0,
                "failed_files": {},
            }

        regenerated_files = 0
        failed_files: dict[str, str] = {}

        for file_name, summary_text in sorted(summaries.items()):
            job = SummaryJob(channel_id=real_channel_id, file_name=file_name)
            try:
                questions = await self._generate_quick_questions_from_summary(
                    summary_text=summary_text,
                    file_name=file_name,
                )
                if not questions:
                    raise ValueError("Quick questions generation returned empty result")

                await self._store_quick_questions(job, questions)
                regenerated_files += 1
            except Exception as e:
                failed_files[file_name] = str(e)
                logger.warning(
                    f"Regenerate quick questions failed for {file_name}: {e}"
                )

        return {
            "channel_id": real_channel_id,
            "deleted_quick_question_keys": deleted_quick_question_keys,
            "total_summaries": len(summaries),
            "regenerated_files": regenerated_files,
            "failed_files": failed_files,
        }

    async def delete(self, channel_id: str, file_name: str) -> None:
        """ลบ summary และ quick questions ออกจาก Redis (เรียกตอน delete document)"""
        job = SummaryJob(channel_id=str(channel_id), file_name=str(file_name))
        try:
            await asyncio.to_thread(
                self.chat_store._redis_client.delete,
                job.redis_key,
                job.quick_questions_redis_key,
            )
            logger.info(
                f"🗑️ Deleted summary + quick questions: {job.redis_key} / "
                f"{job.quick_questions_redis_key}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to delete summary/quick questions for {job.redis_key}: {e}"
            )

    async def is_ready(self, channel_id: str, file_name: str) -> dict[str, Any]:
        """
        ตรวจสถานะความพร้อมของ summary

        Returns:
            {
                "ready": bool,
                "status": str,
                "progress_percent": int,
                "message": str,
                "queue_position": int | None,
                "attempt": int,
                "max_retries": int,
                "redis_key": str,
            }
        """
        job = SummaryJob(channel_id=str(channel_id), file_name=str(file_name))
        summary = await self.get(channel_id, file_name)

        if summary is not None:
            return {
                "ready": True,
                "status": "ready",
                "progress_percent": 100,
                "message": "Summary พร้อมใช้งานแล้ว",
                "queue_position": None,
                "attempt": self._attempts.get(job, 0),
                "max_retries": self.cfg.max_retries,
                "redis_key": job.redis_key,
            }

        if self._active_job == job:
            attempt = max(1, self._attempts.get(job, 1))
            base = 50
            retry_weight = 40
            retry_progress = int(
                ((attempt - 1) / max(self.cfg.max_retries, 1)) * retry_weight
            )
            progress = min(95, base + retry_progress)
            return {
                "ready": False,
                "status": "processing",
                "progress_percent": progress,
                "message": f"กำลังสรุปเอกสาร (attempt {attempt}/{self.cfg.max_retries})",
                "queue_position": 0,
                "attempt": attempt,
                "max_retries": self.cfg.max_retries,
                "redis_key": job.redis_key,
            }

        if job in self._pending:
            queue_position = self._queue_position(job)
            queue_size = max(self._queue.qsize(), 1)
            if queue_position is None:
                progress = 15
                queue_message = "งานอยู่ในคิว"
            else:
                # คิวหน้า ๆ จะได้เปอร์เซ็นต์มากกว่า เพื่อให้ผู้ใช้เห็นความคืบหน้า
                progress = max(10, min(45, 45 - int(((queue_position - 1) / queue_size) * 30)))
                queue_message = f"รอคิวประมวลผล (ลำดับที่ {queue_position})"
            return {
                "ready": False,
                "status": "queued",
                "progress_percent": progress,
                "message": queue_message,
                "queue_position": queue_position,
                "attempt": self._attempts.get(job, 0),
                "max_retries": self.cfg.max_retries,
                "redis_key": job.redis_key,
            }

        failed = self._failed_jobs.get(job)
        if failed:
            return {
                "ready": False,
                "status": "failed",
                "progress_percent": 0,
                "message": (
                    f"สรุปเอกสารไม่สำเร็จหลัง retry ครบ {failed.attempts} ครั้ง: "
                    f"{failed.error}"
                ),
                "queue_position": None,
                "attempt": failed.attempts,
                "max_retries": self.cfg.max_retries,
                "redis_key": job.redis_key,
            }

        nodes = self._get_nodes_for_job(job)
        if nodes:
            return {
                "ready": False,
                "status": "not_queued",
                "progress_percent": 0,
                "message": "พบเอกสารแล้ว แต่ยังไม่ถูกส่งเข้า queue summary",
                "queue_position": None,
                "attempt": 0,
                "max_retries": self.cfg.max_retries,
                "redis_key": job.redis_key,
            }

        return {
            "ready": False,
            "status": "not_found",
            "progress_percent": 0,
            "message": "ไม่พบข้อมูลเอกสารสำหรับสร้าง summary",
            "queue_position": None,
            "attempt": 0,
            "max_retries": self.cfg.max_retries,
            "redis_key": job.redis_key,
        }

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
                self._active_job = job
                await self._process_with_retry(job)
            finally:
                if self._active_job == job:
                    self._active_job = None
                self._queue.task_done()
                self._pending.discard(job)
                self._attempts.pop(job, None)

        logger.info("SummaryWorker loop exited")

    async def _process_with_retry(self, job: SummaryJob) -> None:
        for attempt in range(1, self.cfg.max_retries + 1):
            self._attempts[job] = attempt
            try:
                await self._summarize_and_store(job)
                self._failed_jobs.pop(job, None)
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
                    self._failed_jobs[job] = SummaryFailureInfo(
                        attempts=attempt,
                        error=str(e),
                    )
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

        # 2. Summarize ด้วย primary LLM ก่อน ถ้าพังค่อยใช้ fallback LLM
        try:
            summary_text = await self._summarize_with_llm(nodes, self.llm)
        except Exception as primary_error:
            if not self.fallback_llm:
                raise

            logger.warning(
                f"[yellow]Primary summary LLM failed for {job.file_name}: {primary_error} "
                f"— trying fallback LLM[/]"
            )
            try:
                summary_text = await self._summarize_with_llm(nodes, self.fallback_llm)
                logger.info(
                    f"[bold green]✅ Fallback summary LLM succeeded[/] for {job.file_name}"
                )
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Primary and fallback summary LLM failed for {job.file_name}. "
                    f"primary_error={primary_error}; fallback_error={fallback_error}"
                ) from fallback_error

        # 3. เก็บลง Redis (ถ้าไม่ตั้ง TTL จะใช้ key แบบไม่หมดอายุ)
        if self.cfg.redis_ttl_seconds is None:
            await asyncio.to_thread(
                self.chat_store._redis_client.set,
                job.redis_key,
                summary_text,
            )
        else:
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

        if self.cfg.quick_questions_enabled:
            try:
                quick_questions = await self._generate_quick_questions_from_summary(
                    summary_text=summary_text,
                    file_name=job.file_name,
                )
                if quick_questions:
                    await self._store_quick_questions(job, quick_questions)
            except Exception as e:
                # ไม่ให้ quick question ล้มแล้วลากให้ summary flow fail
                logger.warning(
                    f"Quick question generation failed for {job.file_name}: {e}"
                )

    async def _summarize_with_llm(self, nodes: list, llm: Any) -> str:
        summary_index = SummaryIndex(nodes)
        engine = summary_index.as_query_engine(
            llm=llm,
            response_mode="tree_summarize",
            use_async=True,
        )
        result = await engine.aquery(self.cfg.summarize_prompt)
        summary_text = str(result).strip()
        if not summary_text:
            raise ValueError("Summarization returned empty result")
        return summary_text

    async def _store_quick_questions(self, job: SummaryJob, questions: list[str]) -> None:
        payload = json.dumps(questions, ensure_ascii=False)
        if self.cfg.redis_ttl_seconds is None:
            await asyncio.to_thread(
                self.chat_store._redis_client.set,
                job.quick_questions_redis_key,
                payload,
            )
        else:
            await asyncio.to_thread(
                self.chat_store._redis_client.setex,
                job.quick_questions_redis_key,
                self.cfg.redis_ttl_seconds,
                payload,
            )

        logger.info(
            f"[bold green]✅ Quick questions stored[/] — key={job.quick_questions_redis_key} "
            f"({len(questions)} items)"
        )

    async def _generate_quick_questions_from_summary(
        self,
        summary_text: str,
        file_name: str,
    ) -> list[str]:
        summary_text = summary_text.strip()
        if not summary_text:
            return []

        trimmed_summary = summary_text[: self.cfg.quick_questions_max_summary_chars]
        prompt = self.cfg.quick_questions_prompt.format(
            count=self.cfg.quick_questions_count,
            file_name=file_name,
            summary=trimmed_summary,
        )

        try:
            raw = await self.llm.acomplete(prompt)
            questions = self._extract_quick_questions(str(raw).strip())
        except Exception as primary_error:
            if not self.fallback_llm:
                raise

            logger.warning(
                f"[yellow]Primary quick-question LLM failed for {file_name}: {primary_error} "
                f"— trying fallback LLM[/]"
            )
            raw_fallback = await self.fallback_llm.acomplete(prompt)
            questions = self._extract_quick_questions(str(raw_fallback).strip())

        if not questions and self.fallback_llm:
            raw_fallback = await self.fallback_llm.acomplete(prompt)
            questions = self._extract_quick_questions(str(raw_fallback).strip())

        if not questions:
            raise ValueError("Quick questions generation returned empty result")

        return questions[: self.cfg.quick_questions_count]

    def _extract_quick_questions(self, raw_text: str) -> list[str]:
        """รองรับทั้ง JSON array และ plain-text list จาก LLM"""
        parsed_json = self._parse_json_array(raw_text)
        if parsed_json:
            return parsed_json

        lines = []
        for line in raw_text.splitlines():
            clean = re.sub(r"^\s*(?:[-*•]|\d+[\.)])\s*", "", line).strip()
            if clean:
                lines.append(clean)

        deduped: list[str] = []
        seen: set[str] = set()
        for item in lines:
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)

        return deduped

    @staticmethod
    def _parse_json_array(raw_text: str) -> list[str]:
        candidate = raw_text.strip()

        # รองรับกรณีโมเดลครอบด้วย markdown code fence
        if candidate.startswith("```"):
            candidate = re.sub(r"^```[a-zA-Z]*\n", "", candidate)
            candidate = candidate.removesuffix("```").strip()

        start = candidate.find("[")
        end = candidate.rfind("]")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                deduped: list[str] = []
                seen: set[str] = set()
                for item in parsed:
                    text = str(item).strip()
                    if not text or text in seen:
                        continue
                    seen.add(text)
                    deduped.append(text)
                return deduped
        except Exception:
            return []

        return []

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

    async def _redis_delete_keys_by_pattern(self, pattern: str) -> int:
        try:
            keys: list[bytes] = await asyncio.to_thread(
                self.chat_store._redis_client.keys, pattern
            )
            if not keys:
                return 0

            decoded_keys = [key.decode() for key in keys]
            deleted_count = await asyncio.to_thread(
                self.chat_store._redis_client.delete, *decoded_keys
            )
            return int(deleted_count or 0)
        except Exception as e:
            logger.warning(f"Redis delete by pattern failed (pattern={pattern}): {e}")
            return 0

    def _queue_position(self, job: SummaryJob) -> int | None:
        """คืนลำดับของ job ในคิว (เริ่มที่ 1) ถ้าอยู่ในคิวจริง"""
        try:
            snapshot = list(self._queue._queue)  # noqa: SLF001 - ใช้เพื่อ diagnostics ภายใน
            return snapshot.index(job) + 1
        except ValueError:
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