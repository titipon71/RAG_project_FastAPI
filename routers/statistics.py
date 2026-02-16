from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, cast, Date
from sqlalchemy.ext.asyncio import AsyncSession

from core.enums import RoleUser, RoleChannel
from core.security import get_current_user
from db.session import get_db
from db.models.user import User
from db.models.chat import Chats
from db.models.channel import Channel

router = APIRouter()


# ============================================================
#                  STATISTICS ROUTES
# ============================================================
@router.get("/questions/stats/daily", tags=["Statistics"])
async def number_of_questions_asked_per_day(
    start_date: date | None = Query(None, description="วันเริ่มต้น (YYYY-MM-DD)"),
    end_date: date | None = Query(None, description="วันสิ้นสุด (YYYY-MM-DD)"),
    year: int | None = Query(None, ge=2000, le=2100, description="ปีที่ต้องการ (เช่น 2025)"),
    month: int | None = Query(None, ge=1, le=12, description="เดือนที่ต้องการ (1-12)"),
    day: int | None = Query(None, ge=1, le=31, description="วันที่ต้องการ (1-31)"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="เฉพาะแอดมินเท่านั้น")

    # Group by Date (ตัดเวลาทิ้ง เอาแค่วันที่)
    stmt = (
        select(
            cast(Chats.created_at, Date).label("date"), 
            func.count().label("count")
        )
    )
    
    # --- Date Range Filter ---
    if start_date:
        stmt = stmt.where(cast(Chats.created_at, Date) >= start_date)
    if end_date:
        stmt = stmt.where(cast(Chats.created_at, Date) <= end_date)
    
    # --- Year / Month / Day Filter ---
    if year:
        stmt = stmt.where(func.year(Chats.created_at) == year)
    if month:
        stmt = stmt.where(func.month(Chats.created_at) == month)
    if day:
        stmt = stmt.where(func.day(Chats.created_at) == day)
    
    stmt = stmt.group_by(cast(Chats.created_at, Date)).order_by(cast(Chats.created_at, Date))

    result = await db.execute(stmt)
    data = result.all()
    
    # แปลงผลลัพธ์เป็น List of Dict
    return [
        {"date": row.date, "count": row.count} 
        for row in data
    ]
    
@router.get("/users/stats/daily", tags=["Statistics"])
async def number_of_active_users_per_day(
    start_date: date | None = Query(None, description="วันเริ่มต้น (YYYY-MM-DD)"),
    end_date: date | None = Query(None, description="วันสิ้นสุด (YYYY-MM-DD)"),
    year: int | None = Query(None, ge=2000, le=2100, description="ปีที่ต้องการ (เช่น 2025)"),
    month: int | None = Query(None, ge=1, le=12, description="เดือนที่ต้องการ (1-12)"),
    day: int | None = Query(None, ge=1, le=31, description="วันที่ต้องการ (1-31)"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="เฉพาะแอดมินเท่านั้น")

    # Group by Date (ตัดเวลาทิ้ง เอาแค่วันที่)
    stmt = (
        select(
            cast(Chats.created_at, Date).label("date"), 
            func.count(func.distinct(Chats.users_id)).label("active_users")
        )
    )
    
    # --- Date Range Filter ---
    if start_date:
        stmt = stmt.where(cast(Chats.created_at, Date) >= start_date)
    if end_date:
        stmt = stmt.where(cast(Chats.created_at, Date) <= end_date)
    
    # --- Year / Month / Day Filter ---
    if year:
        stmt = stmt.where(func.year(Chats.created_at) == year)
    if month:
        stmt = stmt.where(func.month(Chats.created_at) == month)
    if day:
        stmt = stmt.where(func.day(Chats.created_at) == day)
    
    stmt = stmt.group_by(cast(Chats.created_at, Date)).order_by(cast(Chats.created_at, Date))

    result = await db.execute(stmt)
    data = result.all()
    
    # แปลงผลลัพธ์เป็น List of Dict
    return [
        {"date": row.date, "active_users": row.active_users} 
        for row in data
    ]

@router.get('/channels/pending/count', tags=["Statistics"])
async def channel_pending_count(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="เฉพาะแอดมินเท่านั้น")

    stmt = select(func.count()).select_from(Channel).where(Channel.status == RoleChannel.pending)
    pending_count = await db.scalar(stmt)  # คืนค่าเป็น int

    return {"Channel_pending_count": int(pending_count or 0)}

@router.get('/channels/public/count', tags=["Statistics"])
async def channel_public_count(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="เฉพาะแอดมินเท่านั้น")

    stmt = select(func.count()).select_from(Channel).where(Channel.status == RoleChannel.public)
    public_count = await db.scalar(stmt)  # คืนค่าเป็น int

    return {"Channel_public_count": int(public_count or 0)}

@router.get('/channels/private/count', tags=["Statistics"])
async def channel_private_count(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="เฉพาะแอดมินเท่านั้น")

    stmt = select(func.count()).select_from(Channel).where(Channel.status == RoleChannel.private)
    private_count = await db.scalar(stmt)  # คืนค่าเป็น int
    return {"Channel_private_count": int(private_count or 0)}
