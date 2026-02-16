import logging
import secrets
import uuid
import asyncio
from datetime import datetime
from typing import List, Tuple

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.enums import RoleChannel, RoleUser
from core.hashids import encode_id, decode_id
from core.security import get_current_user
from core.api_key_security import get_api_key_context, hash_key
from db.session import get_db
from db.models.user import User
from db.models.channel import Channel
from db.models.api_key import ApiKey
from schemas.api_key import ApiKeyCreate, ApiKeyRevoke, ApiKeyResponse, ApiKeyListResponse
from schemas.chat import ExternalChatRequest
from rag_enginex import rag_engine

logger = logging.getLogger("uvicorn.error")

router = APIRouter()


# ============================================================
#                  PUBLIC API ROUTES
# ============================================================
@router.post("/auth/api-keys", response_model=ApiKeyResponse, tags=["Public API"])
async def create_api_key(
    payload: ApiKeyCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    logger.info(f"Start create_api_key | user_id={current_user.users_id}")

    try:
        real_channel_id = None

        # ===== ตรวจสอบ Channel =====
        if payload.channel_id:
            real_channel_id = decode_id(payload.channel_id)

            stmt = (
                select(Channel, ApiKey)
                .outerjoin(ApiKey, ApiKey.channel_id == Channel.channels_id)
                .where(
                    Channel.channels_id == real_channel_id,
                    Channel.created_by == current_user.users_id
                )
            )

            result = await db.execute(stmt)
            channel_with_key = result.first()

            if not channel_with_key:
                raise HTTPException(
                    status_code=403,
                    detail="ไม่มีสิทธิ์สร้าง API Key สำหรับ Channel นี้"
                )

            _, existing_key = channel_with_key

            if existing_key:
                raise HTTPException(
                    status_code=400,
                    detail="มี API Key สำหรับ Channel นี้อยู่แล้ว"
                )

        # ===== สร้าง API Key =====
        raw_key = "sk-" + secrets.token_urlsafe(32)
        hashed = hash_key(raw_key)

        new_key = ApiKey(
            user_id=current_user.users_id,
            channel_id=real_channel_id,
            key_hash=hashed,
            name=payload.name
        )

        db.add(new_key)
        await db.flush()
        await db.refresh(new_key)

        logger.info(f"API key created | id={new_key.key_id}")

        return {
            "key_id": new_key.key_id,
            "name": new_key.name,
            "channel_id": payload.channel_id,
            "key_secret": raw_key,
            "created_at": new_key.created_at or datetime.now()
        }

    except HTTPException:
        # ปล่อย HTTPException เดิมออกไป (ไม่ต้อง log เป็น 500)
        raise

    except Exception as e:
        logger.error(
            f"Error creating API Key | user_id={current_user.users_id} | error={str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )


@router.post("/auth/api-keys/revoke", status_code=204, tags=["Public API"])
async def revoke_api_key(
    payload: ApiKeyRevoke,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # หา key ใน DB
    stmt = select(ApiKey).where(
        ApiKey.key_id == payload.key_id,
        ApiKey.user_id == current_user.users_id
    )
    result = await db.execute(stmt)
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=404, detail="API Key not found")
    await db.delete(api_key)
    
    message = {
        "message": "API Key revoked successfully"
    }
    
    return message

@router.get("/auth/api-keys/list", response_model=List[ApiKeyListResponse], tags=["Public API"])
async def list_api_keys(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        stmt = select(ApiKey).where(ApiKey.user_id == current_user.users_id)
        result = await db.execute(stmt)
        keys = result.scalars().all()

        response = []
        for key in keys:
            response.append(ApiKeyListResponse(
                key_id=key.key_id,
                name=key.name,
                channel_id=encode_id(key.channel_id) if key.channel_id else None,
                key_hint=f"*********{key.key_hash[-4:]}",  # Key จริงที่ยังไม่ Hash (ไม่ส่งกลับใน list)
                created_at=key.created_at or datetime.now()
            ))

        return response

    except Exception as e:
        # จะ log e ไว้ก็ได้ ถ้ามี logger
        logger.error(f"Error listing API keys: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch API keys"
        )



@router.post("/api/v1/chat/completions", tags=["Public API"])
async def public_chat_api(
    payload: ExternalChatRequest,
    db: AsyncSession = Depends(get_db),
    # รับค่าเป็น Tuple จาก Dependency ใหม่
    auth_context: Tuple[User, ApiKey] = Depends(get_api_key_context) 
):
    # Unpack ข้อมูลออกมา
    current_user, current_api_key = auth_context

    # แปลง Channel ID
    real_channel_id = decode_id(payload.channel_id)
    if not real_channel_id:
        raise HTTPException(status_code=404, detail="Invalid Channel ID")

    # ดึงข้อมูล Channel จริงจาก DB
    stmt = select(Channel).where(Channel.channels_id == real_channel_id)
    res = await db.execute(stmt)
    channel = res.scalar_one_or_none()
    
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    # =========================================================
    # 🎯 NEW LOGIC: ตรวจสอบสิทธิ์ (Authorization & Scope)
    # =========================================================
    
    # กรณี 1: Scoped Key (Key นี้ถูกสร้างมาเพื่อห้องนี้โดยเฉพาะ)
    if current_api_key.channel_id is not None:
        # ถ้า Key ระบุห้อง แต่ห้องที่ขอไม่ตรงกัน -> ผิด!
        if current_api_key.channel_id != real_channel_id:
            logger.warning(f"Key Abuse: Key for channel {current_api_key.channel_id} tried to access {real_channel_id}")
            raise HTTPException(status_code=403, detail="This API Key is not valid for the requested channel")
        
        # ถ้าตรงกัน = ผ่านเลย (เพราะคนสร้าง Key คือเจ้าของห้องอยู่แล้ว)

    # กรณี 2: Master Key (Key ไม่ระบุห้อง = เข้าได้ทุกห้องที่เป็นของตัวเอง)
    else:
        # ต้องเช็คความเป็นเจ้าของห้อง
        is_owner = (channel.created_by == current_user.users_id)
        is_admin = (current_user.role == RoleUser.admin)
        is_public_channel = (channel.status == RoleChannel.public)

        if not (is_owner or is_admin or is_public_channel):
             raise HTTPException(status_code=403, detail="Access denied: You do not own this channel")

    # =========================================================
    # ส่วนการทำงาน RAG เดิม
    # =========================================================

    last_user_msg = payload.messages[-1]["content"]
    
    # สร้าง Session ID สำหรับ Redis
    if payload.conversation_id:
        # แยก Session ตาม Key ID ด้วย เพื่อไม่ให้ชนกันถ้าระบบซับซ้อนขึ้น
        redis_session_key = f"api:{current_user.users_id}:{real_channel_id}:{payload.conversation_id}"
    else:
        redis_session_key = f"api_temp:{uuid.uuid4()}"

    try:
        result = await asyncio.to_thread(
            rag_engine.aquery, 
            question=last_user_msg, 
            channel_id=real_channel_id, 
            sessions_id=redis_session_key 
        )
    except Exception as e:
        logger.error(f"RAG Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal AI Error")

    return {
        "id": payload.conversation_id or f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": "qwen3:1.7b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result["answer"]
            },
            "finish_reason": "stop"
        }],
        "usage": result["usage"]
    }
