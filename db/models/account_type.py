from sqlalchemy import String, ForeignKey, text
from sqlalchemy.dialects.mysql import INTEGER as MyInt, ENUM as MyEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from db.base import Base
import db.models.file_size
import db.models.user


class AccountType(Base):
    __tablename__ = "account_type"

    account_type_id: Mapped[int] = mapped_column("account_type_id", MyInt(), primary_key=True, autoincrement=True)
    type_name: Mapped[str] = mapped_column("type_name", String(50), unique=True, nullable=False)
    file_size_id: Mapped[int] = mapped_column("file_size_default_id", MyInt(), ForeignKey("file_size_default.file_size_default_id"), nullable=False)
    
    users = relationship("User", back_populates="account_type_rel")
    file_size = relationship("FileSize", back_populates="account_type_rel")