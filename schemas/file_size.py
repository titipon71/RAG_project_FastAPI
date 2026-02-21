from schemas.base import ORMBase


# --- File Size Schemas ---
class FileSizeCreate(ORMBase):
    size: int
    
class FileSizeUpdate(ORMBase):
    id: int
    size: int
    
