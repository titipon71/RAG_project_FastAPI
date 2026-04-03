import json
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Path, Body
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from core.enums import RoleChannel, RoleUser
from core.hashids import decode_id
from core.security import get_current_user, get_optional_current_user
from db.session import SessionLocal, get_db
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
@router.post("/sessions/ollama-reply", tags=["Chat & AI"])
async def Talking_with_Ollama_from_document(
    payload: ChatRequest = Body(...),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    print(f"🔍 [1] Payload ID: {payload.sessions_id} (Type: {type(payload.sessions_id)})")
    print(f"👤 [2] Current User: {current_user.users_id if current_user else 'Guest (None)'}")

    # --- 1. เตรียมข้อมูลพื้นฐาน ---
    async with SessionLocal() as db:
        real_session_id = payload.sessions_id
        
        
        print(f"🆔 [3] Decoded ID: {real_session_id}")

        # สร้าง Query
        stmt = (
            select(Sessions)
            .options(joinedload(Sessions.channel))
            .where(Sessions.sessions_id == real_session_id)
        )

        # DEBUG: เช็คเงื่อนไข User
        if current_user:
            stmt = stmt.where(Sessions.user_id == current_user.users_id)
            logger.info(f"🔓 [3] User ID {current_user.users_id} is querying their session.")
            print(f"⚓ [4] Query Mode: Registered User (ID: {current_user.users_id})")
        else:
            stmt = stmt.where(Sessions.user_id.is_(None))
            logger.info("🔒 [3] Guest user is querying session.")
            print(f"⚓ [4] Query Mode: Guest (User ID is NULL)")

        res = await db.execute(stmt)
        sess = res.scalar_one_or_none()

        if not sess:
            # 🚨 จุดที่ทำให้เกิด 403
            print(f"❌ [5] RESULT: Session NOT FOUND in DB (Check ID or Owner!)")
            print("==================== DEBUG END =====================\n")
            raise HTTPException(status_code=403, detail="ไม่พบ Session หรือคุณไม่มีสิทธิ์")

        print(f"✅ [5] RESULT: Session Found! (Channel ID: {sess.channel_id})")
        
        c_id = sess.channel_id
        s_id = sess.sessions_id
        u_id = current_user.users_id if current_user else None
        is_public = sess.channel.status == RoleChannel.public

    # --- 2. สร้าง Generator ---
    async def event_stream():
        print(f"🚀 [Stream] Started for Session: {s_id}")
        full_answer = ""
        try:
            async for event in rag_engine.astream_query(payload.message, c_id, s_id):
                if event.get("type") == "token":
                    token = event.get("token", "")
                    full_answer += token
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            
            print(f"💾 [Stream] Finished. Saving to DB...")
            
            async with SessionLocal() as db_context:
                new_chat = Chats(
                    channels_id=c_id,
                    users_id=u_id,
                    sessions_id=s_id,
                    user_message=payload.message,
                    ai_message=rag_engine._strip_think(full_answer),
                )
                db_context.add(new_chat)
                await db_context.commit()
                print(f"✨ [Stream] Saved Successfully! Chat ID: {new_chat.chat_id}")
                
                yield f"data: {json.dumps({'done': True, 'chat_id': new_chat.chat_id}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"💥 [Stream] ERROR: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    print(f"==================== DEBUG END (Streaming Response Sent) =====================\n")
    return StreamingResponse(event_stream(), media_type="text/event-stream")

 
        


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
