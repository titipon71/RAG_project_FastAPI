from datetime import datetime
from typing import Optional

from sqlalchemy import String, ForeignKey
from sqlalchemy.dialects.mysql import INTEGER as MyInt
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from db.base import Base


class File(Base):
    __tablename__ = "files"

    files_id: Mapped[int] = mapped_column("files_id", MyInt(unsigned=True), primary_key=True, autoincrement=True)
    uploaded_by: Mapped[Optional[int]] = mapped_column("uploaded_by", MyInt(unsigned=True), ForeignKey("users.users_id"), nullable=True)
    channel_id: Mapped[Optional[int]] = mapped_column("channel_id", MyInt(unsigned=True), ForeignKey("channels.channels_id"), nullable=False)
    original_filename: Mapped[str] = mapped_column("original_filename", String(512), nullable=False)
    storage_uri: Mapped[str] = mapped_column("storage_uri", String(1024), nullable=False)
    size_bytes: Mapped[Optional[int]] = mapped_column("size_bytes", MyInt(unsigned=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column("created_at", server_default=func.current_timestamp(), nullable=False)

    channel = relationship("Channel", back_populates="files")
    uploader = relationship("User", back_populates="uploaded_files")