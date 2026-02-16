from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from core.hashids import encode_id
from schemas.base import ORMBase


# --- File Schemas  ---
class FileDetail(ORMBase):
    files_id: str = Field(..., description="Hashed ID")
    original_filename: str
    size_bytes: int
    mime: Optional[str]
    channel_id: str 
    public_url: Optional[str] = None
    created_at: datetime
    
    @field_validator('files_id', mode='before')
    def encode_files_id(cls, v):
        if isinstance(v, int):
            return encode_id(v)
        return v
    
    @field_validator('channel_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int):
            return encode_id(v)
        return v

class FileUploadResponse(BaseModel):
    files: list[FileDetail]

class FileListItem(BaseModel):
    files: list[FileDetail]