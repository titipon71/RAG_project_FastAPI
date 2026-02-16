from datetime import datetime
from typing import Optional

from sqlalchemy import String, ForeignKey, text
from sqlalchemy.dialects.mysql import INTEGER as MyInt, ENUM as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from db.base import Base
from core.enums import RoleChannel


class Channel(Base):
    __tablename__ = "channels"

    channels_id: Mapped[int] = mapped_column("channels_id", MyInt(unsigned=True), primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column("title", String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column("description", String(1000), nullable=True)
    status: Mapped[RoleChannel] = mapped_column("status", SAEnum(RoleChannel), nullable=False, default=RoleChannel.private, server_default=text("'private'"))
    created_by: Mapped[int] = mapped_column("created_by", MyInt(unsigned=True), ForeignKey("users.users_id"), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column("created_at", server_default=func.current_timestamp(), nullable=False)

    files = relationship("File", back_populates="channel", cascade="all, delete-orphan")
    creator = relationship("User", back_populates="channels")
    status_events = relationship("ChannelStatusEvent", back_populates="channel", cascade="all, delete-orphan")
    sessions = relationship("Sessions", back_populates="channel", cascade="all, delete-orphan")