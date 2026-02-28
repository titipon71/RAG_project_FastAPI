from db.base import Base
from sqlalchemy.dialects.mysql import INTEGER as MyInt , BIGINT as MyBigInt
from sqlalchemy.orm import Mapped, mapped_column, relationship
import db.models.user
import db.models.account_type


class FileSize(Base):
    __tablename__ = "file_size_default"
    file_size_default_id: Mapped[int] = mapped_column("file_size_default_id", MyInt(unsigned=True), primary_key=True, autoincrement=True)
    size: Mapped[int] = mapped_column("size", MyBigInt(unsigned=True), nullable=False)

