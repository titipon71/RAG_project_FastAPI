from datetime import datetime
from typing import Optional

from sqlalchemy import String
from sqlalchemy.dialects.mysql import INTEGER as MyInt
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from db.base import Base


class Chats(Base):
    __tablename__ = "chats"
    chat_id: Mapped[int] = mapped_column("chat_id", MyInt(unsigned=True), primary_key=True, autoincrement=True)
    channels_id: Mapped[int] = mapped_column("channels_id", MyInt(unsigned=True), nullable=False)
    users_id: Mapped[int] = mapped_column("users_id", MyInt(unsigned=True), nullable=True)
    sessions_id: Mapped[int] = mapped_column("sessions_id", MyInt(unsigned=True), nullable=False)
    user_message: Mapped[str] = mapped_column("user_message", String(2000), nullable=False)
    ai_message: Mapped[Optional[str]] = mapped_column("ai_message", String(2000), nullable=True)
    created_at: Mapped[datetime] = mapped_column("created_at", server_default=func.current_timestamp(), nullable=False)