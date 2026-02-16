from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from core.enums import RoleChannel, ModerationDecision
from core.hashids import encode_id
from schemas.base import ORMBase


# --- Admin / Moderation ---
class ModerationResponse(ORMBase):
    channels_id: str
    old_status: RoleChannel
    current_status: str = Field(validation_alias='new_status') # Map จาก new_status หรือ status
    event_id: int
    message: str = "Success" # Default msg
    
    @field_validator('channels_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int): return encode_id(v)
        return v
    
class AdminDecisionIn(BaseModel):
    approve: bool
    reason: Optional[str] = None

class AdminDecisionOut(ORMBase):
    channels_id: str
    decision: Optional[ModerationDecision]
    status_after: RoleChannel = Field(validation_alias='new_status') # Trick mapping
    event_id: int
    decided_by: Optional[int]
    decided_at: Optional[datetime]
    message: str = "Processed"

    @field_validator('channels_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int): return encode_id(v)
        return v

class UserRequestChannelStatusEventResponse(ORMBase):
    event_id: int
    channel_id: str
    channel_title: str
    old_status: RoleChannel
    new_status: RoleChannel
    requested_by: int
    decided_by: Optional[int] = None 
    decision: Optional[ModerationDecision] = None   
    decision_reason: Optional[str] = None
    decided_at: Optional[datetime] = None
    created_at: datetime

    @field_validator('channel_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int): return encode_id(v)
        return v

    class Config:
        from_attributes = True