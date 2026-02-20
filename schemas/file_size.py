from schemas.base import ORMBase


# --- File Size Schemas ---
class FileSizeCreate(ORMBase):
    id: int 
    size: int
    
