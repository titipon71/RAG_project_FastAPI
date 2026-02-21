from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr

from core.enums import RoleUser
from schemas.base import ORMBase

# --- SSO User Schemas ---
class SSOUserInfo(BaseModel):
    sso_access_token: str


# --- User Schemas ---
class UserCreate(BaseModel):
    username: str
    name: str
    password: str
    email: Optional[EmailStr] = None

class UserOut(ORMBase):
    users_id: int
    username: str
    name: str
    email: Optional[EmailStr] = None
    role: RoleUser
    account_type: Optional[str] = None
    created_at: datetime

class UserUpdate(BaseModel):
    username: Optional[str] = None
    name: Optional[str] = None
    email: Optional[EmailStr] = None

class UserPasswordUpdate(BaseModel):
    old_password: str
    new_password: str
    confirm_password: str

class UserRoleUpdate(BaseModel):
    users_id: int
    new_role: RoleUser
    secret_key: str    