from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey
from sqlalchemy.dialects.mysql import INTEGER as MyInt
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from db.base import Base


class Sessions(Base):
    __tablename__ = "sessions"
    sessions_id: Mapped[int] = mapped_column("sessions_id", MyInt(unsigned=True), primary_key=True, autoincrement=True)
    channel_id: Mapped[int] = mapped_column("channel_id", MyInt(unsigned=True), ForeignKey("channels.channels_id"), nullable=False)
    user_id: Mapped[Optional[int]] = mapped_column("user_id", MyInt(unsigned=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column("created_at", server_default=func.current_timestamp(), nullable=False)
    
    channel = relationship("Channel", back_populates="sessions")