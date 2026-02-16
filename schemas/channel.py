from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field, field_validator

from core.enums import RoleChannel
from core.hashids import encode_id
from schemas.base import ORMBase
from schemas.file import FileDetail


# --- Channel Schemas ---
class ChannelCreate(BaseModel):
    title: str = Field(..., description="ชื่อ Channel" , min_length=1, max_length=255)
    description: Optional[str] = Field(None, description="คำอธิบาย Channel", max_length=1000)

class ChannelResponse(ORMBase):
    channels_id: str
    title: str
    description: Optional[str] = None
    created_by: int
    status: RoleChannel
    
    @field_validator('channels_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int): return encode_id(v)
        return v

class ChannelOut(ORMBase):
    channels_id: str
    title: str
    description: Optional[str]
    status: RoleChannel
    created_at: datetime
    # แก้จาก List[dict] เป็น List[FileDetail] เพื่อให้ validator ทำงาน
    files: List[FileDetail] 
    
    @field_validator('channels_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int): return encode_id(v)
        return v

class ChannelUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None

class ChannelUpdateResponse(ORMBase):
    title: str
    description: Optional[str] = None
    
class ChannelOneResponse(ORMBase):
    channels_id: str
    title: str
    description: Optional[str] = None
    status: RoleChannel
    created_by_id: int = Field(validation_alias='created_by') # Map field DB
    created_by_name: str = "Unknown" # ต้องจัดการใน Router หรือใช้ property
    created_at: datetime
    file_count: int = 0
    files: List[FileDetail]
    
    @field_validator('channels_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int): return encode_id(v)
        return v

    # @field_validator('created_by_id', mode='before')
    # def encode_creator_id(cls, v):
    #     if isinstance(v, int):
    #         return encode_id(v)
    #     return v
    
# --- List Items ---
class ChannelListPendingItem(ChannelOneResponse):
    pass

class ChannelListPublicItem(ChannelOneResponse):
    pass

class ChannelListAllItem(ChannelOneResponse):
    pass

class ChannelUpdateStatus(ORMBase):
    channels_id: str
    status: RoleChannel
    @field_validator('channels_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int): return encode_id(v)
        return v