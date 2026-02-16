from datetime import datetime
from typing import Optional, List, Annotated

from pydantic import BaseModel, field_validator, WithJsonSchema

from core.hashids import encode_id, decode_id
from schemas.base import ORMBase


class ChatRequest(BaseModel):
    sessions_id: Annotated[
        int, 
        WithJsonSchema({
            "type": "string", 
            "example": "string", 
            "description": "Session ID ที่ถูก Hash มาแล้ว"
        })
    ]
    message: str
    
    @field_validator('sessions_id', mode='before')
    def decode_session_id_val(cls, v):
        # ถ้าส่งมาเป็น String (จาก JSON) ให้ Decode
        if isinstance(v, str):
            real_id = decode_id(v)
            if real_id is None: 
                raise ValueError("Invalid Session ID")
            return real_id
        return v

class chatHistoryItem(ORMBase):
    chat_id: int
    channels_id: str
    users_id: int
    sessions_id: str # ควรเป็น str (hash) ถ้าอยากปกปิด
    user_message: str
    ai_message: Optional[str] = None
    created_at: datetime
    
    @field_validator('channels_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int): return encode_id(v)
        return v
    
    @field_validator('sessions_id', mode='before')
    def encode_session_id(cls, v):
        if isinstance(v, int): return encode_id(v)
        return v
    
class ExternalChatRequest(BaseModel):
    channel_id: str  # Hashed Channel ID
    messages: List[dict] # [{"role": "user", "content": "..."}]

    conversation_id: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "channel_id": "AbCd123",
                    "conversation_id": "client-app-user-1234",
                    "messages": [
                        {
                            "role": "user",
                            "content": "ช่วยสรุปสาระสำคัญของเอกสารใน Channel นี้ให้หน่อย"
                        }
                    ]
                }
            ]
        }
    }