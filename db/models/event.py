from datetime import datetime
from typing import Optional

from sqlalchemy import String, ForeignKey, Boolean
from sqlalchemy.dialects.mysql import INTEGER as MyInt, ENUM as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from db.base import Base
from core.enums import RoleChannel, ModerationDecision


class ChannelStatusEvent(Base):
    __tablename__ = "channel_status_events"

    event_id: Mapped[int] = mapped_column(
        "event_id",
        MyInt(unsigned=True),
        primary_key=True,
        autoincrement=True,
    )

    channel_id: Mapped[int] = mapped_column(
        "channel_id",
        MyInt(unsigned=True),
        ForeignKey(
            "channels.channels_id",
            ondelete="CASCADE",   
            onupdate="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    old_status: Mapped[RoleChannel] = mapped_column(
        "old_status",
        SAEnum(RoleChannel),
        nullable=False,
    )
    new_status: Mapped[RoleChannel] = mapped_column(
        "new_status",
        SAEnum(RoleChannel),
        nullable=False,
    )

    requested_by: Mapped[Optional[int]] = mapped_column(
        "requested_by",
        MyInt(unsigned=True),
        ForeignKey(
            "users.users_id",
            ondelete="SET NULL",
            onupdate="CASCADE",
        ),
        nullable=True,
    )

    decided_by: Mapped[Optional[int]] = mapped_column(
        "decided_by",
        MyInt(unsigned=True),
        ForeignKey(
            "users.users_id",
            ondelete="SET NULL",
            onupdate="CASCADE",
        ),
        nullable=True,
    )

    decision: Mapped[Optional[ModerationDecision]] = mapped_column(
        "decision",
        SAEnum(ModerationDecision),
        nullable=True,
    )

    decision_reason: Mapped[Optional[str]] = mapped_column(
        "decision_reason",
        String(1000),
        nullable=True,
    )
    
    is_read: Mapped[bool] = mapped_column(
        "is_read",
        Boolean,
        nullable=False,
        default=False,
        server_default="0"
    )

    created_at: Mapped[datetime] = mapped_column(
        "created_at",
        server_default=func.current_timestamp(),
        nullable=False,
    )
    decided_at: Mapped[Optional[datetime]] = mapped_column(
        "decided_at",
        nullable=True,
    )

    channel = relationship("Channel", back_populates="status_events")
    requester = relationship("User", foreign_keys=[requested_by])
    approver  = relationship("User", foreign_keys=[decided_by])