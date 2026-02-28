from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, computed_field

from core.enums import RoleUser
from db.models.file_size import FileSize
from schemas.base import ORMBase

# --- SSO User Schemas ---
class SSOUserInfo(BaseModel):
    sso_access_token: str


# --- User Schemas ---
class UserCreate(BaseModel):
    username: str
    name: str
    password: str

class UserOutV2(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    users_id: int
    username: str
    name: str
    role: RoleUser
    account_type: Optional[str] = None

    file_size_byte: Optional[int] = None 

    created_at: datetime

class UserOut(ORMBase):
    users_id: int
    username: str
    name: str
    role: RoleUser
    account_type: Optional[str] = None
    created_at: datetime

class UserUpdate(BaseModel):
    username: Optional[str] = None
    name: Optional[str] = None
    
class UserFileSizeUpdate(BaseModel):
    users_id: int
    file_size_byte: int = Field(..., ge=0, description="ขนาดไฟล์ที่ update (หน่วยเป็น byte)")

class UserPasswordUpdate(BaseModel):
    old_password: str
    new_password: str
    confirm_password: str

class UserRoleUpdate(BaseModel):
    users_id: int
    new_role: RoleUser
    secret_key: str    