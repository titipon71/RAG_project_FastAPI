
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from db.models.file_size import FileSize
from db.models.user import User


async def change_user_file_size(
    db: AsyncSession,
    size: int,
    user_id: int
):

    stmt = select(User).where(User.users_id == user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    if not user:
        raise ValueError("User not found")

    user.file_size_custom = size
    await db.flush()
