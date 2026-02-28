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
    role: Mapped[RoleUser] = mapped_column("role", MyEnum(RoleUser), nullable=False, server_default=text("'user'"))
    file_size_custom: Mapped[Optional[int]] = mapped_column("file_size_custom", MyInt(unsigned=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column("created_at", server_default=func.current_timestamp(), nullable=False)

    channels = relationship("Channel", back_populates="creator")
    uploaded_files = relationship("File", back_populates="uploader")
    account_type_rel = relationship("AccountType", back_populates="users")
    
    def get_max_file_size(self) -> Optional[int]:
        if self.file_size_custom is not None:
            return self.file_size_custom

        if (
            self.account_type_rel
            and self.account_type_rel.file_size_default is not None
        ):
            return self.account_type_rel.file_size_default

        return None