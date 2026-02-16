from datetime import datetime
from typing import Optional

from sqlalchemy import String, ForeignKey, Boolean
from sqlalchemy.dialects.mysql import INTEGER as MyInt
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from db.base import Base


class ApiKey(Base):
    __tablename__ = "api_keys"

    key_id: Mapped[int] = mapped_column(MyInt(unsigned=True), primary_key=True, autoincrement=True)
    
    # ผูกกับ User เจ้าของ Key
    user_id: Mapped[int] = mapped_column(MyInt(unsigned=True), ForeignKey("users.users_id"), nullable=False)
    
    channel_id: Mapped[Optional[int]] = mapped_column(MyInt(unsigned=True), ForeignKey("channels.channels_id"), nullable=True)
    
    # เก็บ Key ที่ Hash แล้ว
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    
    # ชื่อเรียก Key เช่น "My Chatbot A"
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # เอาไว้ปิด/เปิด Key นี้
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    created_at: Mapped[datetime] = mapped_column(server_default=func.current_timestamp())

    owner = relationship("User", backref="api_keys")
    channel = relationship("Channel", backref="api_keys")