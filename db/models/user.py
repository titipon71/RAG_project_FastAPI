from datetime import datetime
from typing import Optional

from sqlalchemy import String, text
from sqlalchemy.dialects.mysql import INTEGER as MyInt, ENUM as MyEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from db.base import Base
from core.enums import RoleUser


class User(Base):
    __tablename__ = "users"

    users_id: Mapped[int] = mapped_column(
        "users_id", MyInt(unsigned=True), primary_key=True, autoincrement=True,
    )
    username: Mapped[str] = mapped_column("username", String(255), nullable=False)
    name: Mapped[str] = mapped_column("name", String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column("hashed_password", String(255), nullable=False)
    email: Mapped[Optional[str]] = mapped_column("email", String(320), unique=True, nullable=True)
    account_type: Mapped[str] = mapped_column("account_type", String(50), nullable=True)
    file_size: Mapped[int] = mapped_column("file_size", MyInt(unsigned=True), nullable=False)
    role: Mapped[RoleUser] = mapped_column("role", MyEnum(RoleUser), nullable=False, server_default=text("'user'"))
    created_at: Mapped[datetime] = mapped_column("created_at", server_default=func.current_timestamp(), nullable=False)

    channels = relationship("Channel", back_populates="creator")
    uploaded_files = relationship("File", back_populates="uploader")
    file_size = relationship("FileSize", back_populates="users")