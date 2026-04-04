import asyncio
import logging
import traceback
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status, Request
from fastapi.responses import StreamingResponse
from jose import JWTError, jwt
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core import sse_manager
from core.enums import RoleUser
from core.security import get_current_user
from db.session import get_db
from db.models.user import User
from db.models.event import ChannelStatusEvent
from schemas.event import EventsAsReadRequest
from schemas.moderation import UserRequestChannelStatusEventResponse


logger = logging.getLogger("uvicorn.error")

router = APIRouter()
import os
from dotenv import load_dotenv

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"

# ============================================================
#                 EVENT ROUTES
# ============================================================
@router.get("/events/list/{user_id}", response_model=List[UserRequestChannelStatusEventResponse], tags=["Events & Moderation"])
async def get_channel_status_events_by_user(
    user_id: int = Path(..., description="ID ของผู้ใช้งาน"),
    skip: int = Query(0, ge=0, description="ข้ามจำนวนรายการ (Pagination)"),
    limit: int = Query(10, ge=1, le=10, description="จำนวนสูงสุดที่ต้องการดึง"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # ตรวจสอบสิทธิ์ (ส่วนนี้ไม่ต้อง try/except เพราะถ้า fail คือเจตนาของเรา)
    if current_user.users_id != user_id:
        logger.warning(f"Unauthorized access attempt by UserID: {current_user.users_id} targeting UserID: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="ไม่มีสิทธิ์ดำเนินการเพื่อดูเหตุการณ์เหล่านี้"
        )

    try:
        # --- เริ่มส่วนที่อาจเกิด Error (เช่น DB connection, Query ผิด) ---
        
        # ดึงข้อมูล (ใช้ selectinload เพื่อประสิทธิภาพตามที่แนะนำไปรอบก่อน)
        stmt = (
            select(ChannelStatusEvent)
            .options(selectinload(ChannelStatusEvent.channel))
            .where(ChannelStatusEvent.requested_by == user_id)
            .where(ChannelStatusEvent.soft_delete == False)
            .order_by(ChannelStatusEvent.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        
        result = await db.execute(stmt)
        events = result.scalars().all()

        # แปลงข้อมูล
        event_list = []
        for ev in events:
            response_item = UserRequestChannelStatusEventResponse(
                event_id=ev.event_id,
                channel_id=ev.channel_id,
                channel_title=ev.channel.title if ev.channel else "Unknown Channel",
                old_status=ev.old_status,
                new_status=ev.new_status,
                requested_by=ev.requested_by,
                decided_by=ev.decided_by,
                decision=ev.decision,
                decision_reason=ev.decision_reason,
                created_at=ev.created_at,
                decided_at=ev.decided_at,
                is_read=ev.is_user_read,
                is_admin_read=ev.is_admin_read,
            )
            event_list.append(response_item)
        
        return event_list

    except Exception as e:
        # --- ส่วนจัดการเมื่อเกิด Error ---
        
        # 1. Log error ตัวจริงลงระบบ (User ไม่เห็นส่วนนี้)
        # exc_info=True จะปริ้นท์ Stack Trace ยาวๆ ออกมาให้เราแก้บั๊ก
        logger.error(f"Error fetching events for UserID {user_id}: {str(e)}", exc_info=True)
        
        # 2. ส่ง Error ทั่วไปกลับไปหา User (เพื่อความปลอดภัย ไม่ควรบอก Error จริง)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="เกิดข้อผิดพลาดภายในระบบ ไม่สามารถดึงข้อมูลได้ในขณะนี้"
        )


def get_user_id_from_token(token: str) -> int:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token payload")

        return int(user_id)

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ✅ SSE endpoint
# @router.get("/events/stream/{user_id}", tags=["Events & Moderation"])
# async def stream_notifications(
#     request: Request,
#     user_id: int = Path(...),
#     token: str = Query(...),  # 👈 รับ token จาก query
# ):
#     print("=== SSE CONNECT ===")
#     print("user_id:", user_id)
#     print("query:", dict(request.query_params))
#     print("===================")

#     # 🔐 auth
#     current_user_id = get_user_id_from_token(token)

#     # 🔒 authorization
#     if current_user_id != user_id:
#         raise HTTPException(status_code=403, detail="ไม่มีสิทธิ์")

#     async def event_generator():
#         q = sse_manager.connect(user_id)

#         try:
#             while True:
#                 # ❌ client ปิด connection หรือยัง
#                 if await request.is_disconnected():
#                     print(f"Client disconnected: user_id={user_id}")
#                     break

#                 try:
#                     # ⏳ รอ event 30 วิ
#                     data = await asyncio.wait_for(q.get(), timeout=30.0)

#                     yield (
#                         "event: notification\n"
#                         f"data: {data}\n\n"
#                     )

#                 except asyncio.TimeoutError:
#                     # 💓 heartbeat กัน connection ตาย
#                     yield "event: ping\ndata: {}\n\n"

#         except asyncio.CancelledError:
#             print(f"Cancelled: user_id={user_id}")

#         finally:
#             sse_manager.disconnect(user_id, q)
#             print(f"Disconnected cleanup: user_id={user_id}")

#     return StreamingResponse(
#         event_generator(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive",
#             "X-Accel-Buffering": "no",  # สำคัญถ้าใช้ Nginx
#         },
#     )
    
@router.post("/events/read", status_code=204, tags=["Events & Moderation"])
async def events_as_read(
    payload: EventsAsReadRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        if payload.type not in ["user", "admin"]:
            raise HTTPException(status_code=400, detail="Invalid type. Must be 'user' or 'admin'.")
        
        if payload.type == "user":
            stmt = (
                update(ChannelStatusEvent)
                .where(ChannelStatusEvent.event_id == payload.event_id)
                .where(ChannelStatusEvent.requested_by == current_user.users_id)
                .values(is_user_read=True) 
            )        
        elif payload.type == "admin":
            stmt = (
                update(ChannelStatusEvent)
                .where(ChannelStatusEvent.event_id == payload.event_id)
                .where(ChannelStatusEvent.requested_by == current_user.users_id)
                .values(is_admin_read=True) 
            )
        
        result = await db.execute(stmt)
        
        if result.rowcount == 0:
            print(f"DEBUG: No rows updated. Event ID {payload.event_id} not found or user mismatch.")
            # ถ้าไม่เจอ ถือว่าไม่มีอะไรผิดพลาดร้ายแรง แค่หาไม่เจอ
            raise HTTPException(status_code=404, detail="Event not found")
            
        print("DEBUG: Update successful!")
        
    except Exception as e:
        # พิมพ์ Error ออกมาดู
        print("################ ERROR DETAILS ################")
        print(traceback.format_exc()) 
        print("###############################################")
        raise HTTPException(status_code=500, detail=str(e)) # ส่ง error text กลับไปที่ client ด้วย

    return

@router.post("/events/soft-delete", status_code=204, tags=["Events & Moderation"])
async def events_soft_delete(
    payload: EventsAsReadRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        stmt = (
            update(ChannelStatusEvent)
            .where(ChannelStatusEvent.event_id == payload.event_id)
            .where(ChannelStatusEvent.requested_by == current_user.users_id)
            .where(ChannelStatusEvent.soft_delete == False)
            .values(soft_delete=True)
        )

        result = await db.execute(stmt)

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Event not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error soft-deleting event_id={payload.event_id} for user_id={current_user.users_id}: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="เกิดข้อผิดพลาดภายในระบบ ไม่สามารถลบข้อมูลได้ในขณะนี้"
        )

    return