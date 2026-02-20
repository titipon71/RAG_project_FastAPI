from db.base import Base
from sqlalchemy.dialects.mysql import INTEGER as MyInt
from sqlalchemy.orm import Mapped, mapped_column, relationship

class FileSize(Base):
    __tablename__ = "file_size"
    file_size_id: Mapped[int] = mapped_column("file_size_id", MyInt(unsigned=True), primary_key=True, autoincrement=True)
    size: Mapped[int] = mapped_column("size", MyInt(unsigned=True), nullable=False)

    users = relationship("User", back_populates="file_size")
