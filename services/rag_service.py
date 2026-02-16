from typing import List

from rag_enginex import rag_engine

# ============================================================
#                      RAG / AI HELPERS
# ============================================================

async def call_ai(messages: List[dict], channel_id: int, session_id: int) -> dict:
    last_user_msg = None
    for m in reversed(messages):
        if m["role"] == "user":
            last_user_msg = m["content"]
            break
    if last_user_msg is None:
        last_user_msg = "สรุปข้อมูลจากฐานเอกสารให้หน่อย"

    result = await rag_engine.aquery(last_user_msg, channel_id, session_id)
    return result