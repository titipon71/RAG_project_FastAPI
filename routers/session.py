import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Path, Body
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from core.enums import RoleChannel, RoleUser
from core.hashids import decode_id
from core.security import get_current_user, get_optional_current_user
from db.session import get_db
from db.models.user import User
from db.models.channel import Channel
from db.models.session import Sessions
from db.models.chat import Chats
from schemas.session import SessionCreate, SessionResponse
from schemas.chat import ChatRequest, chatHistoryItem
from services.session_service import get_owned_session
from services.rag_service import call_ai
from rag_enginex import rag_engine

logger = logging.getLogger("uvicorn.error")

router = APIRouter()


# ============================================================
#                  SESSION ROUTES
# ============================================================
@router.post("/session", status_code=201, response_model=SessionResponse, tags=["Sessions"])
async def create_session(
    payload: SessionCreate,
    db: AsyncSession = Depends(get_db),
    # 👇 ใช้ Optional ตรงนี้ด้วย
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    # 1) หา channel
    result = await db.execute(
        select(Channel).where(Channel.channels_id == payload.channel_id)
    )
    channel = result.scalar_one_or_none()
    if channel is None:
        raise HTTPException(status_code=404, detail="ไม่พบ Channel")

    user_id = current_user.users_id if current_user else None
    
    # 2) เช็คสิทธิ์
    if user_id:
        # กรณี User: เข้าได้ถ้าเป็นเจ้าของ, แอดมิน หรือ ห้อง public
        is_admin = (current_user.role == RoleUser.admin)
        is_owner = (channel.created_by == user_id)
        
        if channel.status in (RoleChannel.private, RoleChannel.pending) and not (is_owner or is_admin):
            raise HTTPException(status_code=403, detail="ไม่มีสิทธิ์เข้าถึง Channel นี้")
    else:
        # กรณี Guest: เข้าได้เฉพาะ Public เท่านั้น
        if channel.status != RoleChannel.public:
            raise HTTPException(
                status_code=401, 
                detail="กรุณาเข้าสู่ระบบเพื่อใช้งาน Channel ส่วนตัว"
            )

    # 3) สร้าง session (user_id จะเป็น None ถ้าเป็น Guest)
    new_session = Sessions(
        channel_id=payload.channel_id,
        user_id=user_id, 
    )
    db.add(new_session)
    await db.flush()
    await db.refresh(new_session)
    return new_session

@router.delete("/session/delete/{session_id}", status_code=204, tags=["Sessions"])
async def delete_session(
    session_id: str = Path(..., title="The hashed session ID"), # 1. เปลี่ยนรับเป็น String (Hash)
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    real_session_id = decode_id(session_id)
    
    result = await db.execute(select(Sessions).where(Sessions.sessions_id == real_session_id))
    session = result.scalar_one_or_none()
    
    if session is None:
        raise HTTPException(status_code=404, detail="ไม่พบ Session")
    
    isAdmin = (current_user.role == RoleUser.admin)
    isOwner = (session.user_id == current_user.users_id)
    
    if not isAdmin and not isOwner:
        raise HTTPException(status_code=403, detail="ไม่มีสิทธิ์ดำเนินการลบ Session นี้")
    
    await db.delete(session)
    
    rag_engine.clear_session_history(session.sessions_id) 
    
    return


# ============================================================
#                  CHAT + AI ROUTES
# ============================================================
@router.post("/sessions/ollama-reply", status_code=201, tags=["Chat & AI"])
async def Talking_with_Ollama_from_document(
    payload: ChatRequest = Body(...),
    db: AsyncSession = Depends(get_db),
    # รับได้ทั้ง User และ None (Guest)
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    try:
        real_session_id = payload.sessions_id
        if isinstance(real_session_id, str):
             real_session_id = decode_id(real_session_id)

        # -------------------------------------------------------
        # STEP 1: ค้นหา Session และ Join Channel ในคำสั่งเดียว
        # -------------------------------------------------------
        stmt = (
            select(Sessions)
            .options(joinedload(Sessions.channel)) # 👈 ใช้ joinedload ได้แล้ว
            .where(Sessions.sessions_id == real_session_id)
        )

        # Filter เจ้าของ Session
        if current_user:
            stmt = stmt.where(Sessions.user_id == current_user.users_id)
        else:
            stmt = stmt.where(Sessions.user_id.is_(None)) # Guest ต้องเป็น NULL

        res = await db.execute(stmt)
        sess = res.scalar_one_or_none()

        if sess is None:
            raise HTTPException(status_code=403, detail="ไม่พบ Session หรือคุณไม่มีสิทธิ์")

        # -------------------------------------------------------
        # STEP 2: ตรวจสอบสถานะ Channel (เข้าถึงผ่าน sess.channel ได้เลย)
        # -------------------------------------------------------
        # ถ้าเป็น Guest เข้าได้เฉพาะ Public Channel
        if not current_user:
             if sess.channel.status != RoleChannel.public:
                 raise HTTPException(status_code=401, detail="Guest ใช้ได้เฉพาะ Public Channel เท่านั้น")

        # -------------------------------------------------------
        # STEP 3: Logic การบันทึกและตอบกลับ
        # -------------------------------------------------------
        sender_id = current_user.users_id if current_user else None

        # 1. เรียก AI ก่อน
        ai_messages = [{"role": "user", "content": payload.message}]
        ai_result = await call_ai(ai_messages, sess.channel_id, sess.sessions_id)

        # 2. บันทึกทั้ง User Message และ AI Message ลงในแถวเดียว
        new_chat = Chats(
            channels_id=sess.channel_id,
            users_id=sender_id,
            sessions_id=sess.sessions_id,
            user_message=payload.message,      # ใส่คำถาม
            ai_message=ai_result["answer"],     # ใส่คำตอบ
        )
        db.add(new_chat)
        await db.flush()
        await db.refresh(new_chat)

        return {
            "user_message": {
                "chat_id": new_chat.chat_id,
                "   ": new_chat.user_message,
                "created_at": new_chat.created_at,
            },
            "ai_message": { 
                "message": new_chat.ai_message,
            },
            "token_usage": ai_result["usage"]
        }

    except HTTPException as he:
        logger.error(f"HTTPException: {he.detail}", exc_info=True)
        return JSONResponse(
            status_code=he.status_code,
            content={"message": "แจ้ง backend ด้วยจ้า", "detail": he.detail}
        )
    except Exception as e:
        logger.error(f"Ollama Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/history", response_model=List[chatHistoryItem], tags=["Chat & AI"])
async def get_chat_history(session_id: str = Path(...),
                           db: AsyncSession = Depends(get_db),
                           current_user: User = Depends(get_current_user)
                           ):
    real_session_id = decode_id(session_id)
    if real_session_id is None:
         raise HTTPException(status_code=404, detail="รูปแบบ Session ID ไม่ถูกต้อง")
     
    # 1) ตรวจว่า session นี้เป็นของ user นี้จริงไหม
    owned_sess = await get_owned_session(db, real_session_id, current_user.users_id)
    if owned_sess is None:
        # ไม่ใช่ของเขา หรือไม่มีอยู่
        raise HTTPException(status_code=403, detail="ไม่ใช่ Session ของคุณ")

    # 2) ดึงประวัติ chat ทั้งหมดใน session นี้
    result = await db.execute(
        select(Chats)
        .where(Chats.sessions_id == real_session_id)
        .order_by(Chats.created_at.asc())
    )
    chat_rows = result.scalars().all()

    history = []
    for row in chat_rows:
        history.append(chatHistoryItem(
            chat_id=row.chat_id,
            channels_id=row.channels_id,
            users_id=row.users_id,
            sessions_id=row.sessions_id,
            user_message=row.user_message,
            ai_message=row.ai_message,
            created_at=row.created_at,
        ))

    return history
