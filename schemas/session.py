from datetime import datetime
from typing import Optional, Annotated

from pydantic import BaseModel, field_validator, WithJsonSchema

from core.hashids import encode_id, decode_id
from schemas.base import ORMBase


# --- Session & Chat ---
class SessionCreate(BaseModel):
    channel_id: Annotated[
        int, 
        WithJsonSchema({"type": "string", "example": "string"}) 
    ]
    
    @field_validator('channel_id', mode='before')
    def decode_channel_id(cls, v):
        if isinstance(v, str):
            decoded = decode_id(v) 
            if decoded is None:
                raise ValueError("Channel ID ไม่ถูกต้อง")
            return decoded
        return v

class SessionResponse(ORMBase):
    sessions_id: str
    channel_id: str
    user_id: Optional[int] = None
    created_at: datetime
    
    @field_validator('sessions_id', mode='before')
    def encode_session_id(cls, v):
        if isinstance(v, int): return encode_id(v) 
        return v    

    @field_validator('channel_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int): return encode_id(v)
        return v