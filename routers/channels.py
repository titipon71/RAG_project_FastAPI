import logging
import os
import pathlib
import secrets
import shutil
from datetime import datetime, timezone
import traceback
from typing import List, Optional
from unittest import result

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path
from fastapi.responses import JSONResponse
from nanoid import generate
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload
from starlette.datastructures import UploadFile

from core.config import settings
from core.enums import RoleChannel, ModerationDecision, RoleUser
from core.hashids import encode_id, decode_id
from core.security import get_current_user
from db.models.account_type import AccountType
from db.session import get_db
from db.models.user import User
from db.models.channel import Channel
from db.models.file import File
from db.models.event import ChannelStatusEvent
from schemas.channel import (
    ChannelCreate, ChannelResponse, ChannelOut, ChannelUpdate,
    ChannelUpdateResponse, ChannelOneResponse, ChannelListPendingItem,
    ChannelListPublicItem, ChannelListAllItem, ChannelUpdateStatus, ChannelFileSizeBalanceResponse
)
from schemas.file import FileDetail
from schemas.moderation import ModerationResponse, AdminDecisionIn, AdminDecisionOut
from services.channel_service import get_latest_pending_event
from rag_enginex import rag_engine

logger = logging.getLogger("uvicorn.error")

router = APIRouter()

UPLOAD_ROOT = settings.upload_root

# ============================================================
#                  CHANNEL & FILE SETTINGS
# ============================================================

# ตรวจไฟล์ (optional)
MAX_SIZE_PER_FILE = 1 * 1024 * 1024 * 1024  # 1 GB
ALLOW_MIME = {"application/pdf",
              "text/plain"}

try:
    import magic
    def sniff_mime(path: pathlib.Path) -> str:
        return magic.from_file(str(path), mime=True) or "application/octet-stream"
except Exception:
    def sniff_mime(path: pathlib.Path) -> str:
        # fallback อย่างน้อยใช้สกุลไฟล์
        return "application/pdf" if path.suffix.lower() == ".pdf" else "application/octet-stream"

def _build_storage_path(channel_id: int, filename: str) -> tuple[pathlib.Path, str]:
    """คืน (abs_path, relative_path) โดย relative ใช้เก็บใน DB"""
    ext = pathlib.Path(filename or "").suffix.lower()
    uid = secrets.token_hex(16)  # uuid ก็ได้
    rel = pathlib.Path(str(channel_id)) / f"{uid}{ext}"
    abs_path = UPLOAD_ROOT / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    return abs_path, str(rel).replace("\\", "/")  # normalize

async def _save_upload_atomic(uf: UploadFile, final_path: pathlib.Path, max_size: int) -> int:
    """เขียนไฟล์ลง temp แล้ว atomic replace ไปยังปลายทาง"""
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    size_bytes = 0
    async with aiofiles.open(tmp_path, "wb") as f:
        while True:
            chunk = await uf.read(1024 * 1024)
            if not chunk:
                break
            size_bytes += len(chunk)
            if max_size is not None and size_bytes > max_size:
                await uf.close()
                raise HTTPException(status_code=413, detail=f"ไฟล์มีขนาดใหญ่เกินไป: {uf.filename}")
            await f.write(chunk)
    # replace แบบ atomic
    os.replace(tmp_path, final_path)
    return size_bytes


# ============================================================
#                  CHANNEL ROUTES
# ============================================================
@router.post("/channels", status_code=201, response_model=ChannelResponse, tags=["Channels"])
async def create_channel(
    channel_in: ChannelCreate,  
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    new_channel = Channel(
        title=channel_in.title,
        description=channel_in.description,
        created_by=current_user.users_id,
        search_key=generate(size=10),  # สร้าง search_key แบบสุ่ม (หรือจะใช้ hashids จาก ID ก็ได้)
        status=RoleChannel.private
    )

    db.add(new_channel)
    await db.flush()
    await db.refresh(new_channel) 
    return new_channel



import traceback
from fastapi import HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload, selectinload

@router.get("/channels/{channel_id}", response_model=ChannelOut, tags=["Channels"])
async def get_channel_details(
    channel_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        logger.info(f"Request channel_id={channel_id}, user_id={current_user.users_id}")

        decoded_channel_id = decode_id(channel_id)
        logger.debug(f"Decoded channel_id={decoded_channel_id}")

        stmt = (
            select(Channel)
            .options(
                joinedload(Channel.creator).joinedload(User.account_type_rel),
                selectinload(Channel.files)
            )
            .where(Channel.channels_id == decoded_channel_id)
        )

        result = await db.execute(stmt)
        channel = result.scalars().unique().one_or_none()

        if channel is None:
            logger.warning(f"Channel not found: {decoded_channel_id}")
            raise HTTPException(status_code=404, detail="ไม่พบช่อง Channel")

        is_private_like = channel.status in (RoleChannel.private, RoleChannel.pending)
        is_owner = channel.created_by == current_user.users_id
        is_admin = current_user.role == RoleUser.admin

        if is_private_like and not (is_owner or is_admin):
            logger.warning(
                f"Permission denied: user={current_user.users_id}, channel={decoded_channel_id}"
            )
            raise HTTPException(status_code=403, detail="ไม่มีสิทธิ์เข้าถึง channel")

        creator = channel.creator

        max_size = max_size = creator.get_max_file_size() if creator else None

        file_list = []
        for f in channel.files:
            try:
                mime_type = sniff_mime(UPLOAD_ROOT / f.storage_uri)
            except Exception as mime_err:
                logger.error(f"MIME error for file {f.files_id}: {mime_err}")
                mime_type = "unknown"

            item = {
                "files_id": f.files_id,
                "original_filename": f.original_filename,
                "size_bytes": f.size_bytes,
                "created_at": f.created_at,
                "mime": mime_type,
                "channel_id": f.channel_id
            }

            if channel.status == RoleChannel.public:
                item["public_url"] = f"/static/uploads/{f.storage_uri}"
            else:
                item["public_url"] = None

            file_list.append(item)

        return {
            "channels_id": channel.channels_id,
            "title": channel.title,
            "description": channel.description,
            "status": channel.status,
            "maximum_file_size": max_size,
            "created_at": channel.created_at,
            "files": file_list
        }

    except HTTPException as http_err:
        logger.warning(f"HTTPException: {http_err.detail}")
        raise http_err

    except Exception as e:
        logger.exception("🔥 ERROR in get_channel_details")
        
        # logger.exception จะ log traceback ให้อัตโนมัติ

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.delete("/channels/{channel_id}", tags=["Channels"])
async def delete_channel(channel_id: str, db: AsyncSession = Depends(get_db),current_user: User = Depends(get_current_user)):
    decoded_channel_id = decode_id(channel_id)
    # ดึง channel ตาม id
    result = await db.execute(
        select(Channel).where(Channel.channels_id == decoded_channel_id)
    )
    channel = result.scalar_one_or_none()

    if channel is None:
        # ถ้าไม่เจอ ให้คืน 404
        raise HTTPException(status_code=404, detail="ไม่พบ Channel")

    if current_user.role != RoleUser.admin and channel.created_by != current_user.users_id:
        raise HTTPException(
            status_code=403,
            detail="ไม่มีสิทธิ์ลบ Channel นี้"
        ) 
    
    flie_row_result = await db.execute(select(File).where(File.channel_id == channel.channels_id))
    flie_row = flie_row_result.scalars().all()
    
    # ลบเอกสารใน RAG (Chroma) ทั้งหมดที่เกี่ยวข้องกับไฟล์ใน channel นี้
    for file in flie_row:            
        try:
            # ลบเอกสารใน RAG (Chroma)
            rag_engine.delete_documents_by_file_id(file.files_id)
        except Exception as e:
            print(f"[RAG] failed to delete documents for file_id {file.files_id}: {e}")
    
    # ลบไฟล์ในช่องทั้ง DB + ดิสก์
    for fr in flie_row:
        try:
            (UPLOAD_ROOT / fr.storage_uri).unlink(missing_ok=True)
        except Exception:
            pass
        await db.delete(fr)
    await db.delete(channel)
    return {"message": "Channel deleted successfully"}

@router.put("/channels/{channel_id}", response_model=ChannelUpdateResponse, tags=["Channels"])
async def update_channel(
    channel_id: str,
    payload: ChannelUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    decoded_channel_id = decode_id(channel_id)
    stmt = select(Channel).where(Channel.channels_id == decoded_channel_id)
    result = await db.execute(stmt)
    
    channels = result.scalars().one_or_none() 

    if channels is None:
        raise HTTPException(status_code=404, detail="ไม่พบ Channel")
    
    isNotOwner = channels.created_by != current_user.users_id
    isNotAdmin = current_user.role != RoleUser.admin
    
    if isNotOwner and isNotAdmin:
        raise HTTPException(status_code=403, detail="เฉพาะเจ้าของหรือแอดมินเท่านั้นที่สามารถอัปเดต channel ได้")

    channels.title = payload.title
    channels.description = payload.description

    await db.flush() 
    
    await db.refresh(channels)
    
    return channels

@router.post("/channels/{channel_id}/request-public", response_model=ModerationResponse, status_code=201, tags=["Events & Moderation"])
async def request_make_public(
    channel_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1) โหลด channel
    decoded_channel_id = decode_id(channel_id)
    res = await db.execute(select(Channel).where(Channel.channels_id == decoded_channel_id))
    channel = res.scalar_one_or_none()
    if not channel:
        raise HTTPException(status_code=404, detail="ไม่พบ Channel")

    # 2) ต้องเป็นเจ้าของเท่านั้น
    if channel.created_by != current_user.users_id:
        raise HTTPException(status_code=403, detail="เฉพาะเจ้าของเท่านั้นที่สามารถยื่นคำขอได้")
    # 3) ยื่นคำขอได้เฉพาะเมื่อสถานะเป็น private เท่านั้น
    if channel.status == RoleChannel.public:
        raise HTTPException(status_code=400, detail="Channel เป็นสาธารณะแล้ว")
    if channel.status == RoleChannel.pending:
        raise HTTPException(status_code=409, detail="Channel อยู่ระหว่างรอการอนุมัติ")

    old_status = channel.status
    channel.status = RoleChannel.pending

    # 4) สร้าง event
    event = ChannelStatusEvent(
        channel_id=channel.channels_id,
        old_status=old_status,
        new_status=RoleChannel.public,
        requested_by=current_user.users_id,
        decision=None,
    )
    db.add(event)
    await db.flush()
    await db.refresh(channel)
    await db.refresh(event)

    return ModerationResponse(
        channels_id=channel.channels_id,
        old_status=old_status,
        current_status=channel.status,
        event_id=event.event_id,
        message="Request submitted. Waiting for admin approval."
    )

@router.post("/channels/{channel_id}/cancel-request", response_model=ModerationResponse, status_code=201, tags=["Events & Moderation"])
async def cancel_request_make_public(
    channel_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1) Decode และตรวจสอบ ID
    decoded_channel_id = decode_id(channel_id)
    if decoded_channel_id is None:
        raise HTTPException(status_code=404, detail="Channel ID ไม่ถูกต้อง")

    # 2) โหลด channel
    # ใช้ with_for_update เพื่อกัน Race Condition
    res = await db.execute(
        select(Channel).where(Channel.channels_id == decoded_channel_id).with_for_update()
    )
    channel = res.scalar_one_or_none()
    
    if not channel:
        raise HTTPException(status_code=404, detail="ไม่พบ Channel")

    # 3) ต้องเป็นเจ้าของเท่านั้น
    if channel.created_by != current_user.users_id:
        raise HTTPException(status_code=403, detail="เฉพาะเจ้าของเท่านั้นที่สามารถยกเลิกคำขอได้")
    
    # 4) ยกเลิกได้เฉพาะเมื่อสถานะเป็น pending เท่านั้น
    if channel.status != RoleChannel.pending:
        raise HTTPException(status_code=400, detail="Channel ไม่ได้อยู่ระหว่างรอการอนุมัติ")

    # 5) ค้นหา Event ใบเดิมที่ค้างอยู่ (The Pending Ticket)
    # ใช้ฟังก์ชัน get_latest_pending_event ที่คุณประกาศไว้แล้ว
    pending_event = await get_latest_pending_event(db, decoded_channel_id)

    if not pending_event:
        # กรณี Data หลุด (channel เป็น pending แต่หา event ไม่เจอ)
        # ให้ force update channel กลับเป็น private ได้เลย แต่แจ้งเตือนหน่อย
        channel.status = RoleChannel.private
        await db.flush()
        return ModerationResponse(
            channels_id=channel_id,
            old_status=RoleChannel.pending, # สมมติ
            new_status=RoleChannel.private,
            event_id=0,
            message="Status reverted (Event log not found)."
        )

    # 6) --- [จุดสำคัญ] Update Event เดิม แทนการสร้างใหม่ ---
    now = datetime.now(timezone.utc)
    
    # อัปเดตข้อมูลใน Event เดิม
    pending_event.decided_by = current_user.users_id  # ระบุว่าเจ้าของเป็นคนตัดสินใจเอง
    pending_event.decision = ModerationDecision.rejected # ใช้ rejected ในความหมายว่า "คำขอตกไป"
    pending_event.decision_reason = "Cancelled by user" # ใส่เหตุผลให้ชัดเจน
    pending_event.decided_at = now
    
    # 7) เปลี่ยนสถานะ Channel กลับ
    old_status_msg = channel.status
    channel.status = RoleChannel.private

    await db.flush()
    await db.refresh(channel)
    await db.refresh(pending_event)

    return ModerationResponse(
        channels_id=channel.channels_id,
        old_status=old_status_msg,
        new_status=channel.status,
        event_id=pending_event.event_id,
        message="Request cancelled. Channel reverted to private."
    )

@router.post("/channels/{channel_id}/moderate-public", response_model=AdminDecisionOut, tags=["Events & Moderation"])
async def approved_rejected_public_request(
    channel_id: str,
    payload: AdminDecisionIn,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="เฉพาะแอดมินเท่านั้น")
    
    decoded_channel_id = decode_id(channel_id)
    res = await db.execute(select(Channel)
                           .where(Channel.channels_id == decoded_channel_id)
                            .with_for_update())
    channel = res.scalar_one_or_none()
    if not channel:
        raise HTTPException(status_code=404, detail="ไม่พบ Channel")
    
    if channel.status != RoleChannel.pending:
        raise HTTPException(status_code=400, detail="Channel ถูกดำเนินการไปแล้วโดย Admin คนอื่น")

    event_table = await get_latest_pending_event(db, decoded_channel_id)
    if not event_table:
        raise HTTPException(status_code=404, detail="ไม่พบเหตุการณ์คำขอที่รอดำเนินการ")
    
    if event_table.decision is not None:
        raise HTTPException(
            status_code=409, 
            detail="คำขอนี้ถูกดำเนินการไปแล้วโดย Admin คนอื่น"
        )
    
    now =  datetime.now(timezone.utc)
    
    reason = payload.reason
    if payload.approve:
        final_message = reason or "Approved — channel is now PUBLIC."
        channel.status = RoleChannel.public
        event_table.decision = ModerationDecision.approved
    else:
        final_message = reason or "Rejected — channel remains PRIVATE."
        channel.status = RoleChannel.private
        event_table.decision = ModerationDecision.rejected

    event_table.decided_by = current_user.users_id
    event_table.decided_at = now
    event_table.decision_reason = reason

    await db.flush()
    await db.refresh(channel)
    await db.refresh(event_table)

    return AdminDecisionOut(
        channels_id=channel.channels_id,
        decision=event_table.decision,
        status_after=channel.status,
        event_id=event_table.event_id,
        decided_by=event_table.decided_by,
        decided_at=event_table.decided_at,
        message=final_message,
    )

@router.post("/channels/{channel_id}/admin-forced-public", response_model=AdminDecisionOut, tags=["Events & Moderation"])
async def admin_forced_this_channel_to_be_public(
    channel_id: str,
    payload: AdminDecisionIn,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1. Check Admin Permission
    if current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="เฉพาะแอดมินเท่านั้น")
    
    # 2. Get Channel
    real_channel_id = decode_id(channel_id)
    if real_channel_id is None:
        raise HTTPException(status_code=404, detail="Channel ID ไม่ถูกต้อง")
    res = await db.execute(select(Channel).where(Channel.channels_id == real_channel_id))
    channel = res.scalar_one_or_none()
    if not channel:
        raise HTTPException(status_code=404, detail="ไม่พบ Channel")
    
    # 3. Validation: ทำได้เฉพาะช่องที่เป็น Public หรือ Pending เท่านั้น
    if channel.status != RoleChannel.private and channel.status != RoleChannel.pending:
        raise HTTPException(status_code=400, detail="Channel เป็นส่วนตัวแล้ว")
    # เก็บสถานะเดิมไว้ทำ Log
    old_status = channel.status
    now = datetime.now(timezone.utc)
    reason = payload.reason or "Admin forced channel to PUBLIC."

    # 4. Action: เปลี่ยนสถานะเป็น Public
    channel.status = RoleChannel.public
    
    # 5. Create NEW Event Log
    # เราสร้าง Event ใหม่เลย เพราะนี่คือการกระทำใหม่จาก Admin (ไม่ใช่การอนุมัติคำขอเก่า)
    event = ChannelStatusEvent(
        channel_id=channel.channels_id,
        old_status=old_status,
        new_status=RoleChannel.public,
        requested_by=current_user.users_id, # Admin เป็นคนเริ่ม
        decided_by=current_user.users_id,   # Admin เป็นคนตัดสิน
        decision=ModerationDecision.approved,
        decision_reason=reason,
        created_at=now,
        decided_at=now
    )

    db.add(event)
    await db.flush()
    await db.refresh(channel)
    await db.refresh(event)

    return AdminDecisionOut(
        channels_id=channel.channels_id,
        decision=event.decision,
        status_after=channel.status,
        event_id=event.event_id,
        decided_by=event.decided_by,
        decided_at=event.decided_at,
        message=reason,
    )

@router.post("/channels/{channel_id}/admin-forced-private", response_model=AdminDecisionOut, tags=["Events & Moderation"])
async def admin_forced_this_channel_to_be_private(
    channel_id: str,
    payload: AdminDecisionIn,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    real_channel_id = decode_id(channel_id)
    if real_channel_id is None:
        raise HTTPException(status_code=404, detail="Channel ID ไม่ถูกต้อง")
    
    # 1. Check Admin Permission
    if current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="เฉพาะแอดมินเท่านั้น")
    
    # 2. Get Channel
    res = await db.execute(select(Channel).where(Channel.channels_id == real_channel_id))
    channel = res.scalar_one_or_none()
    if not channel:
        raise HTTPException(status_code=404, detail="ไม่พบ Channel")
    
    # 3. Validation: ทำได้เฉพาะช่องที่เป็น Public หรือ Pending เท่านั้น
    if channel.status != RoleChannel.public and channel.status != RoleChannel.pending:
        raise HTTPException(status_code=400, detail="Channel เป็นส่วนตัวแล้ว")
    # เก็บสถานะเดิมไว้ทำ Log
    old_status = channel.status
    now = datetime.now(timezone.utc)
    reason = payload.reason or "Admin forced channel to PRIVATE."

    # 4. Action: เปลี่ยนสถานะเป็น Private
    channel.status = RoleChannel.private
    
    # 5. Create NEW Event Log
    # เราสร้าง Event ใหม่เลย เพราะนี่คือการกระทำใหม่จาก Admin (ไม่ใช่การอนุมัติคำขอเก่า)
    event = ChannelStatusEvent(
        channel_id=channel.channels_id,
        old_status=old_status,
        new_status=RoleChannel.private,
        requested_by=current_user.users_id, # Admin เป็นคนเริ่ม
        decided_by=current_user.users_id,   # Admin เป็นคนตัดสิน
        decision=ModerationDecision.approved,
        decision_reason=reason,
        created_at=now,
        decided_at=now
    )

    db.add(event)
    await db.flush()
    await db.refresh(channel)
    await db.refresh(event)

    return AdminDecisionOut(
        channels_id=channel.channels_id,
        decision=event.decision,
        status_after=channel.status,
        event_id=event.event_id,
        decided_by=event.decided_by,
        decided_at=event.decided_at,
        message=reason,
    )


@router.post("/channels/{channels_id}/owner-set-private", response_model=ModerationResponse, tags=["Events & Moderation"])
async def owner_set_this_channel_private(
    channels_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1. ค้นหา Channel
    decoded_channel_id = decode_id(channels_id)
    res = await db.execute(select(Channel).where(Channel.channels_id == decoded_channel_id))
    channel = res.scalar_one_or_none()
    if not channel:
        raise HTTPException(status_code=404, detail="ไม่พบ Channel")

    # 2. ตรวจสอบสิทธิ์: ต้องเป็นเจ้าของ Channel หรือ Admin เท่านั้น
    if channel.created_by != current_user.users_id and current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="เฉพาะเจ้าของเท่านั้นที่สามารถตั้งค่า Channel เป็นส่วนตัวได้")
    # 3. ตรวจสอบสถานะปัจจุบัน: ถ้าเป็น private อยู่แล้วไม่ต้องทำอะไร
    if channel.status == RoleChannel.private:
        raise HTTPException(status_code=400, detail="Channel เป็นส่วนตัวแล้ว")

    old_status = channel.status
    channel.status = RoleChannel.private

    # 4. บันทึกประวัติการเปลี่ยนสถานะใน ChannelStatusEvent
    event = ChannelStatusEvent(
        channel_id=channel.channels_id,
        old_status=old_status,
        new_status=RoleChannel.private,
        requested_by=current_user.users_id,
        decided_by=current_user.users_id,
        decision=ModerationDecision.approved, # ถือว่าอนุมัติทันทีโดยเจ้าของ
        decision_reason="Changed to private by owner",
        decided_at=datetime.now(timezone.utc)
    )
    db.add(event)
    
    await db.flush()
    await db.refresh(channel)
    await db.refresh(event)

    return ModerationResponse(
        channels_id=channel.channels_id,
        old_status=old_status,
        current_status=channel.status,
        event_id=event.event_id,
        message="Channel is now private."
    )

# @router.put("/channels/admin/debug/force-status", response_model=ChannelUpdateStatus)
# async def update_channel_status(
#     channel_id: int,
#     new_status: RoleChannel = Body(..., embed=True),
#     db: AsyncSession = Depends(get_db),
#     current_user: User = Depends(get_current_user),
# ):
#     result = await db.execute(select(Channel).where(Channel.channels_id == channel_id))
#     channel = result.scalar_one_or_none()
#     if channel is None:
#         raise HTTPException(status_code=404, detail="Channel not found")
#     # เฉพาะ admin เท่านั้นที่แก้ไขได้
#     if current_user.role != RoleUser.admin:
#         raise HTTPException(status_code=403, detail="Admin only")
    
#     channel.status = new_status
#     await db.flush()
#     await db.refresh(channel)
    
#     return ChannelUpdateStatus(
#         channels_id=channel.channels_id,
#         status=channel.status,
#     )

@router.get("/channels/pending/list/", response_model=List[ChannelListPendingItem], tags=["Channels"])
async def list_pending_channels(
    search_by_key_and_name: str | None = Query(None, description="ค้นหาจากชื่อหรือ search_key"),
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Channel)
        .options(
            joinedload(Channel.creator).joinedload(User.account_type_rel), 
            selectinload(Channel.files) 
        )
        .where(Channel.status == RoleChannel.pending)
        .order_by(Channel.created_at.desc())
    )

    if search_by_key_and_name:
        stmt = stmt.where(
            func.lower(Channel.title).contains(search_by_key_and_name.lower()) |
            func.lower(Channel.search_key).contains(search_by_key_and_name.lower())
        )

    stmt = stmt.offset(skip).limit(limit)

    result = await db.execute(stmt)
    channels = result.unique().scalars().all()

    channel_list = []
    for ch in channels:
        
        file_list = [
            {
                "files_id": f.files_id,
                "original_filename": f.original_filename,
                "storage_uri": f.storage_uri,
                "size_bytes": f.size_bytes,
                "created_at": f.created_at,
                "mime": None,
                "channel_id": f.channel_id,  # เพิ่ม channel_id (จำเป็น)    
            }
            for f in ch.files #
        ]
        max_file_size = (
                        ch.creator.get_max_file_size()
                        if ch.creator else None
                        )
                
        channel_list.append(ChannelListPendingItem(
            channels_id=ch.channels_id,
            title=ch.title,
            description=ch.description,
            status=ch.status,
            created_by_id=ch.created_by,
            created_by_name=ch.creator.name if ch.creator else "Unknown", # เข้าถึงผ่าน relationship
            maximum_file_size=max_file_size,
            search_key=ch.search_key,
            created_at=ch.created_at,
            file_count=len(ch.files),
            files=file_list,
        ))
    if not channel_list:
        return JSONResponse(status_code=200, content={"message": "ไม่พบช่องทางที่รอการอนุมัติ (Pending Channels)"})
    return channel_list


@router.get("/channels/public/list/", response_model=List[ChannelListPublicItem], tags=["Channels"])
async def list__channels(
    search_by_key_and_name: str | None = Query(None, description="ค้นหาจากชื่อหรือ search_key"),
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Channel)
        .options(
            joinedload(Channel.creator).joinedload(User.account_type_rel), 
            selectinload(Channel.files) 
        )
        .where(Channel.status == RoleChannel.public)
        .order_by(Channel.created_at.desc())
    )

    if search_by_key_and_name:
        stmt = stmt.where(
            func.lower(Channel.title).contains(search_by_key_and_name.lower()) |
            func.lower(Channel.search_key).contains(search_by_key_and_name.lower())
        )

    stmt = stmt.offset(skip).limit(limit)

    result = await db.execute(stmt)
    channels = result.unique().scalars().all()

    channel_list = []
    for ch in channels:
        
        file_list = [
            {
                "files_id": f.files_id,
                "original_filename": f.original_filename,
                "storage_uri": f.storage_uri,
                "size_bytes": f.size_bytes,
                "created_at": f.created_at,
                "mime": None,
                "channel_id": f.channel_id,  # เพิ่ม channel_id (จำเป็น)    
            }
            for f in ch.files #
        ]
        max_file_size = (
                        ch.creator.get_max_file_size()
                        if ch.creator else None
                        )
                
        channel_list.append(ChannelOneResponse(
            channels_id=ch.channels_id,
            title=ch.title,
            description=ch.description,
            status=ch.status,
            created_by_id=ch.created_by,
            created_by_name=ch.creator.name if ch.creator else "Unknown", # เข้าถึงผ่าน relationship
            maximum_file_size=max_file_size,
            search_key=ch.search_key,
            created_at=ch.created_at,
            file_count=len(ch.files),
            files=file_list,
        ))
    if not channel_list:
        return JSONResponse(status_code=200, content={"message": "ไม่พบช่องทางสาธารณะ (Public Channels)"})   
    return channel_list
    

@router.get("/channels/list/", response_model=List[ChannelOneResponse], tags=["Channels"])
async def list_my_channels(
    search_by_name_and_key: str | None = Query(None, description="ค้นหาจากชื่อหรือ search_key"),
    skip: int = Query(0, ge=0, description="ข้ามจำนวนรายการ (Pagination)"),
    limit: int = Query(100, ge=1, le=150, description="จำนวนรายการต่อหน้า"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        stmt = (
            select(Channel)
            .options(
                selectinload(Channel.creator).selectinload(User.account_type_rel),                
                selectinload(Channel.files) 
            )
            .where(Channel.created_by == current_user.users_id)
            .order_by(Channel.created_at.desc())
        )

        if search_by_name_and_key:
            stmt = stmt.where(
                func.lower(Channel.title).contains(search_by_name_and_key.lower()) |
                func.lower(Channel.search_key).contains(search_by_name_and_key.lower())
            )

        stmt = stmt.offset(skip).limit(limit)

        result = await db.execute(stmt)
        # จุดตายที่ 1: สำหรับ SQLAlchemy Async ถ้ามี joinedload ต้องใช้ .unique()
        channels = result.unique().scalars().all() 

        channel_list = []
        for ch in channels:
            
            # จุดตายที่ 2: การคำนวณค่าใน Loop
            max_size = ch.creator.get_max_file_size() if ch.creator else None

            file_list = [
                {
                    "files_id": f.files_id,
                    "original_filename": f.original_filename,
                    "storage_uri": f.storage_uri,
                    "size_bytes": f.size_bytes,
                    "created_at": f.created_at,
                    "channel_id": f.channel_id,
                    "mime": None,
                }
                for f in ch.files
            ]

            channel_list.append(ChannelOneResponse(
                channels_id=ch.channels_id,
                title=ch.title,
                description=ch.description,
                status=ch.status,
                created_by_id=ch.created_by,
                created_by_name=ch.creator.name if ch.creator else "Unknown",
                created_at=ch.created_at,
                file_count=len(ch.files),
                files=file_list,
                maximum_file_size=max_size,
                search_key=ch.search_key,
            ))
        return channel_list

    except Exception as e:
        # พิมพ์ Error แบบละเอียดลงใน Terminal/Log
        error_traceback = traceback.format_exc()
        logger.error(f"Error in list_my_channels: {str(e)}\n{error_traceback}")
        
        # ส่งรายละเอียดกลับไปที่ Client (เฉพาะตอน Debug)
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Error: {str(e)}" # หรือส่ง error_traceback ถ้าอยากเห็นบน Postman
        )

@router.get("/channels/list/all/", response_model=List[ChannelListAllItem], tags=["Channels"])
async def list_all_channels(
    search_by_name_and_key: str | None = Query(None, description="ค้นหาจากชื่อหรือ search_key"),
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    # current_user: User = Depends(get_current_user),
):
    # if current_user.role != RoleUser.admin:
    #     raise HTTPException(status_code=403, detail="เฉพาะแอดมินเท่านั้น")

    try:
        stmt = (
            select(Channel)
            .options(
                    joinedload(Channel.creator)
                    .joinedload(User.account_type_rel),
                selectinload(Channel.files)
            )
            .order_by(Channel.created_at.desc())
        )

        if search_by_name_and_key:
            stmt = stmt.where(
                func.lower(Channel.title).contains(search_by_name_and_key.lower()) |
                func.lower(Channel.search_key).contains(search_by_name_and_key.lower())
            )

        stmt = stmt.offset(skip).limit(limit)

        result = await db.execute(stmt)
        # จุดตายที่ 1: สำหรับ SQLAlchemy Async ถ้ามี joinedload ต้องใช้ .unique()
        channels = result.unique().scalars().all() 

        channel_list = []
        for ch in channels:
            # Logic การดึงค่าเหมือนที่ทำในหน้า userinfo
            acc_type = ch.creator.account_type_rel if ch.creator else None
            
            # จุดตายที่ 2: การคำนวณค่าใน Loop
            max_size = None
            max_size = ch.creator.get_max_file_size() if ch.creator else None

            file_list = [
                {
                    "files_id": f.files_id,
                    "original_filename": f.original_filename,
                    "storage_uri": f.storage_uri,
                    "size_bytes": f.size_bytes,
                    "created_at": f.created_at,
                    "channel_id": f.channel_id,
                    "mime": None,
                }
                for f in ch.files
            ]

            channel_list.append(ChannelOneResponse(
                channels_id=ch.channels_id,
                title=ch.title,
                description=ch.description,
                status=ch.status,
                created_by_id=ch.created_by,
                created_by_name=ch.creator.name if ch.creator else "Unknown",
                created_at=ch.created_at,
                file_count=len(ch.files),
                files=file_list,
                maximum_file_size=max_size,
                search_key=ch.search_key,
            ))
            
        return channel_list

    except Exception as e:
        # พิมพ์ Error แบบละเอียดลงใน Terminal/Log
        error_traceback = traceback.format_exc()
        logger.error(f"Error in list_all_channels: {str(e)}\n{error_traceback}")
        
        # ส่งรายละเอียดกลับไปที่ Client (เฉพาะตอน Debug)
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Error: {str(e)}" # หรือส่ง error_traceback ถ้าอยากเห็นบน Postman
        )

@router.get('/channels/{channel_id}/storage-usage', response_model=ChannelFileSizeBalanceResponse, tags=["Channels"])
async def get_channel_file_size_balance(
    channel_id: str,
    db: AsyncSession = Depends(get_db)
):
    decoded_id = decode_id(channel_id)
    
    stmt = (
        select(Channel)
        .options(
            selectinload(Channel.creator)
            .selectinload(User.account_type_rel)
        )
        .where(Channel.channels_id == decoded_id)
    )
    result = await db.execute(stmt)
    channel = result.scalar_one_or_none()

    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    sum_stmt = select(func.sum(File.size_bytes)).where(File.channel_id == decoded_id)
    sum_result = await db.execute(sum_stmt)
    used_storage = sum_result.scalar() or 0

    storage_limit = channel.creator.get_max_file_size()
    
    remaining_storage = (storage_limit - used_storage) if storage_limit is not None else None
    
    return ChannelFileSizeBalanceResponse(
        channel_id=channel_id,
        remaining_storage_bytes=remaining_storage,
        used_storage_bytes=used_storage,
        storage_limit_bytes=storage_limit
    )
    
