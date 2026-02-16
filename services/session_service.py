from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models.session import Sessions


async def get_owned_session(
    db: AsyncSession,
    session_id: int,
    user_id: int,
):
    stmt = (
        select(Sessions)
        .where(
            Sessions.sessions_id == session_id,
            Sessions.user_id == user_id,   
        )
    )
    res = await db.execute(stmt)
    return res.scalar_one_or_none()