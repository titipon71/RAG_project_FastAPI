# core/sse_manager.py
import asyncio
from collections import defaultdict

class SSEManager:
    def __init__(self):
        # { user_id: [Queue, Queue, ...] }  รองรับ multi-tab
        self._queues: dict[int, list[asyncio.Queue]] = defaultdict(list)

    def connect(self, user_id: int) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._queues[user_id].append(q)
        return q

    def disconnect(self, user_id: int, q: asyncio.Queue):
        queues = self._queues.get(user_id, [])
        if q in queues:
            queues.remove(q)

    async def push(self, user_id: int, data: dict):
        """เรียกจากที่อื่น (เช่น moderation endpoint) เพื่อ push notification"""
        import json
        for q in self._queues.get(user_id, []):
            await q.put(json.dumps(data, ensure_ascii=False, default=str))

# Singleton
sse_manager = SSEManager()