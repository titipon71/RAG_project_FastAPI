from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# --- Public API Key Schemas ---
class ApiKeyCreate(BaseModel):
    name: str
    channel_id: Optional[str] = None
    

class ApiKeyRevoke(BaseModel):
    key_id: int = Field(..., description="ID ของ API Key ที่ต้องการเพิกถอน")

# Schema สำหรับส่ง Key กลับ (แสดง key_secret แค่ครั้งเดียว)
class ApiKeyResponse(BaseModel):
    key_id: int
    name: str
    channel_id: str | None = None
    key_secret: str # Key จริงที่ยังไม่ Hash
    created_at: datetime

class ApiKeyListResponse(BaseModel):
    key_id: int
    name: str
    channel_id: Optional[str]
    key_hint: str
    created_at: datetime