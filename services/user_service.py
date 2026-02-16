from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models.user import User


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    stmt = select(User).where(User.username == username)
    res = await db.execute(stmt)
    return res.scalar_one_or_none()

async def get_user_by_name(db: AsyncSession, name: str) -> Optional[User]:
    stmt = select(User).where(User.name == name)
    res = await db.execute(stmt)
    return res.scalar_one_or_none()

async def get_user_by_id(db: AsyncSession, uid: int) -> Optional[User]:
    stmt = select(User).where(User.users_id == uid)
    res = await db.execute(stmt)
    return res.scalar_one_or_none()

async def authenticate_user(db: AsyncSession, username: str, password: str) -> Optional[User]:
    from core.security import verify_password  # lazy import to avoid circular
    user = await get_user_by_username(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user