from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, String, text
from sqlalchemy.dialects.mysql import INTEGER as MyInt, ENUM as MyEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from db.base import Base
from core.enums import RoleUser
import db.models.file_size
import db.models.account_type

class User(Base):
    __tablename__ = "users"

    users_id: Mapped[int] = mapped_column("users_id", MyInt(unsigned=True), primary_key=True, autoincrement=True,)
    username: Mapped[str] = mapped_column("username", String(255), nullable=False)
    name: Mapped[str] = mapped_column("name", String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column("hashed_password", String(255), nullable=False)
    account_type_id: Mapped[int] = mapped_column("account_type_id", MyInt(unsigned=True), ForeignKey("account_type.account_type_id"), nullable=True)
    file_size_default_id: Mapped[int] = mapped_column("file_size_default_id", MyInt(unsigned=True),ForeignKey("file_size_default.file_size_default_id"), nullable=False)
    role: Mapped[RoleUser] = mapped_column("role", MyEnum(RoleUser), nullable=False, server_default=text("'user'"))
    created_at: Mapped[datetime] = mapped_column("created_at", server_default=func.current_timestamp(), nullable=False)

    channels = relationship("Channel", back_populates="creator")
    uploaded_files = relationship("File", back_populates="uploader")
    file_size = relationship("FileSize", back_populates="users")
    account_type_rel = relationship("AccountType", back_populates="users")