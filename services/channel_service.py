from typing import Optional

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from core.enums import RoleChannel
from db.models.event import ChannelStatusEvent


async def get_latest_pending_event( db: AsyncSession, channel_id: int) -> Optional[ChannelStatusEvent]:
    stmt = (
        select(ChannelStatusEvent)
        .where(ChannelStatusEvent.channel_id == channel_id,
               ChannelStatusEvent.decision.is_(None),
               ChannelStatusEvent.new_status == RoleChannel.public
               )
        .order_by(desc(ChannelStatusEvent.created_at))
        .limit(1)
        .with_for_update()
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()