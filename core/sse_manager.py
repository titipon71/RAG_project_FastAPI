# core/sse_manager.py
import asyncio
import json
from collections import defaultdict

_SENTINEL = object()  # สัญญาณพิเศษให้ generator หยุด

class SSEManager:
    def __init__(self):
        self._queues: dict[int, list[asyncio.Queue]] = defaultdict(list)
        self._shutdown = False

    def connect(self, user_id: int) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._queues[user_id].append(q)
        return q

    def disconnect(self, user_id: int, q: asyncio.Queue):
        queues = self._queues.get(user_id, [])
        if q in queues:
            queues.remove(q)

    async def push(self, user_id: int, data: dict):
        for q in self._queues.get(user_id, []):
            await q.put(json.dumps(data, ensure_ascii=False, default=str))

    async def shutdown(self):
        """เรียกตอน lifespan shutdown — ส่ง sentinel ให้ทุก queue"""
        self._shutdown = True
        for queues in self._queues.values():
            for q in queues:
                await q.put(_SENTINEL)  # ปลุก generator ที่ block อยู่ทุกตัว

sse_manager = SSEManager()