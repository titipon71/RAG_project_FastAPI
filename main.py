# main.py
# ============================================================
#                      IMPORTS
# ============================================================
import asyncio
import base64
from datetime import datetime, timedelta, timezone
import json
import logging
import shutil
from typing import Optional, AsyncGenerator, List
# from urllib import response
import uuid, pathlib
import aiofiles
from fastapi import Body, FastAPI, APIRouter, Depends, File as FastAPIFile, Form, UploadFile, HTTPException, status, Query, Path , Request, Response
from fastapi.concurrency import asynccontextmanager
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# import httpx
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel , EmailStr, field_validator, Field
from pydantic_settings import BaseSettings
from sqlalchemy import String, desc, func, select ,Enum as SAEnum, ForeignKey, text
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase, joinedload, relationship, selectinload
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import IntegrityError
import enum
from sqlalchemy.dialects.mysql import INTEGER as MyInt, ENUM as MyEnum
from fastapi.middleware.cors import CORSMiddleware
import os, secrets
from llama_index.core import SimpleDirectoryReader
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from hashids import Hashids

from dotenv import load_dotenv

from rag_enginex import rag_engine


# from fastapi_limiter import FastAPILimiter
# from fastapi_limiter.depends import RateLimiter
# from fastapi.responses import JSONResponse


# ============================================================
#                      SETTINGS / CONFIG
# ============================================================

load_dotenv()
class Settings(BaseSettings):
    database_url: str
    secret_key: str = os.getenv("SECRET_KEY")
    access_token_expire_minutes: int = 720
    upload_root: pathlib.Path = pathlib.Path("./uploads")
    TRASH_DIR: pathlib.Path = pathlib.Path("./trash")
    HASH_SALT: str = os.getenv("HASH_SALT")
    MIN_LENGTH: int = int(os.getenv("MIN_LENGTH", 8))
    class Config:
        env_file = ".env"

settings = Settings()

# ============================================================
#                  Hashids INITIALIZATION
# ============================================================
hasher = Hashids(salt=settings.HASH_SALT, min_length=settings.MIN_LENGTH)
def encode_id(hashed_id: int) -> str:
    
    # debug
    print(f"Encoding ID: {hashed_id}")
    
    return hasher.encode(hashed_id)

def decode_id(hashed_id: str) -> Optional[int]:
    if not hashed_id:
        return None
    
    decoded = hasher.decode(hashed_id)
    # debug
    print(f"Decoding hashed ID: {hashed_id} to {decoded}")
    
    if not decoded:
        return None
    return decoded[0]

# ============================================================
#                  DB BASE & ENUMS (SQLAlchemy)
# ============================================================
class Base(DeclarativeBase):
    pass

class RoleUser(str, enum.Enum):
    user = "user"
    admin = "admin"

class RoleChannel(str, enum.Enum):
    public = "public"
    private = "private"
    pending = "pending"

class RoleSender(str, enum.Enum):
    user = "user"
    AI = "AI"

class Theme(str, enum.Enum):
    light = "light"
    dark = "dark"
    
class ModerationDecision(str, enum.Enum):
    approved = "approved"
    rejected = "rejected"


# ============================================================
#                      ORM MODELS (SQLAlchemy)
# ============================================================
class Channel(Base):
    __tablename__ = "channels"

    channels_id: Mapped[int] = mapped_column(
        "channels_id",
        MyInt(unsigned=True),              
        primary_key=True,
        autoincrement=True,
    )

    title: Mapped[str] = mapped_column(
        "title",
        String(255),
        nullable=False,
    )

    description: Mapped[Optional[str]] = mapped_column(
        "description",
        String(1000),
        nullable=True,
    )

    # status: ENUM('public','private') DEFAULT 'private'
    status: Mapped[RoleChannel] = mapped_column(
        "status",
        SAEnum(RoleChannel),                       
        nullable=False,
        default=RoleChannel.private,
        server_default=text("'private'"),      
    )

    created_by: Mapped[int] = mapped_column(
        "created_by",
        MyInt(unsigned=True),
        ForeignKey("users.users_id"),
        nullable=False,
        index=True,
    )
    
    created_at: Mapped[datetime] = mapped_column(
        "created_at",
        server_default=func.current_timestamp(),
        nullable=False,
    )
    
    files = relationship("File", back_populates="channel",cascade="all, delete-orphan")
    creator = relationship("User", back_populates="channels")
    status_events = relationship("ChannelStatusEvent", back_populates="channel",cascade="all, delete-orphan")
    
class User(Base):
    __tablename__ = "users"

    # PK: INT(10) UNSIGNED AUTO_INCREMENT
    users_id: Mapped[int] = mapped_column(
        "users_id",
        MyInt(unsigned=True),              
        primary_key=True,
        autoincrement=True,
    )

    # username: NOT NULL
    username: Mapped[str] = mapped_column(
        "username",
        String(255),
        nullable=False,
    )
    
    # name: UNIQUE, NOT NULL
    name: Mapped[str] = mapped_column(
        "name",
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    hashed_password: Mapped[str] = mapped_column(
        "hashed_password",
        String(255),
        nullable=False,
    )

    # email: UNIQUE, NULL ได้
    email: Mapped[Optional[str]] = mapped_column(
        "email",
        String(320),
        unique=True,
        nullable=True,
    )

    # role: ENUM('user','admin') DEFAULT 'user'
    role: Mapped[RoleUser] = mapped_column(
        "role",
        MyEnum(RoleUser),                       
        nullable=False,
        server_default=text("'user'"),     
    )
    
    theme: Mapped[Theme] = mapped_column(
        "theme",
        MyEnum(Theme),
        nullable=False,
        server_default=text("'light'"),
    )

    created_at: Mapped[datetime] = mapped_column(
        "created_at",
        server_default=func.current_timestamp(),
        nullable=False,
    )
    
    channels = relationship("Channel", back_populates="creator")
    uploaded_files = relationship("File", back_populates="uploader")
class File(Base):
    __tablename__ = "files"
    files_id: Mapped[int] = mapped_column("files_id", MyInt(unsigned=True), primary_key=True, autoincrement=True)
    uploaded_by: Mapped[Optional[int]] = mapped_column("uploaded_by", MyInt(unsigned=True),ForeignKey("users.users_id"), nullable=True)
    channel_id: Mapped[Optional[int]] = mapped_column("channel_id", MyInt(unsigned=True), ForeignKey("channels.channels_id") , nullable=False)
    original_filename: Mapped[str] = mapped_column("original_filename", String(512), nullable=False)
    storage_uri: Mapped[str] = mapped_column("storage_uri", String(1024), nullable=False)
    size_bytes: Mapped[Optional[int]] = mapped_column("size_bytes", MyInt(unsigned=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column("created_at", server_default=func.current_timestamp(), nullable=False)

    channel = relationship("Channel", back_populates="files")
    uploader = relationship("User", back_populates="uploaded_files")
class sessions(Base):
    __tablename__ = "sessions"
    sessions_id: Mapped[int] = mapped_column("sessions_id", MyInt(unsigned=True), primary_key=True, autoincrement=True)
    channel_id: Mapped[int] = mapped_column("channel_id", MyInt(unsigned=True), nullable=False)
    user_id: Mapped[int] = mapped_column("user_id", MyInt(unsigned=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column("created_at", server_default=func.current_timestamp(), nullable=False)

class chats(Base):
    __tablename__ = "chats"
    chat_id: Mapped[int] = mapped_column("chat_id", MyInt(unsigned=True), primary_key=True, autoincrement=True)
    channels_id: Mapped[int] = mapped_column("channels_id", MyInt(unsigned=True), nullable=False)
    users_id: Mapped[int] = mapped_column("users_id", MyInt(unsigned=True), nullable=False)
    sessions_id: Mapped[int] = mapped_column("sessions_id", MyInt(unsigned=True), nullable=False)
    message: Mapped[str] = mapped_column("message", String(2000), nullable=False)
    sender_type: Mapped[RoleSender] = mapped_column("sender_type", MyEnum(RoleSender), nullable=False ,)
    created_at: Mapped[datetime] = mapped_column("created_at", server_default=func.current_timestamp(), nullable=False)


class ChannelStatusEvent(Base):
    __tablename__ = "channel_status_events"

    event_id: Mapped[int] = mapped_column(
        "event_id",
        MyInt(unsigned=True),
        primary_key=True,
        autoincrement=True,
    )

    channel_id: Mapped[int] = mapped_column(
        "channel_id",
        MyInt(unsigned=True),
        ForeignKey(
            "channels.channels_id",
            ondelete="CASCADE",   
            onupdate="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    old_status: Mapped[RoleChannel] = mapped_column(
        "old_status",
        SAEnum(RoleChannel),
        nullable=False,
    )
    new_status: Mapped[RoleChannel] = mapped_column(
        "new_status",
        SAEnum(RoleChannel),
        nullable=False,
    )

    requested_by: Mapped[int] = mapped_column(
        "requested_by",
        MyInt(unsigned=True),
        ForeignKey(
            "users.users_id",
            ondelete="SET NULL",
            onupdate="CASCADE",
        ),
        nullable=True,
    )

    decided_by: Mapped[Optional[int]] = mapped_column(
        "decided_by",
        MyInt(unsigned=True),
        ForeignKey(
            "users.users_id",
            ondelete="SET NULL",
            onupdate="CASCADE",
        ),
        nullable=True,
    )

    decision: Mapped[Optional[ModerationDecision]] = mapped_column(
        "decision",
        SAEnum(ModerationDecision),
        nullable=True,
    )

    decision_reason: Mapped[Optional[str]] = mapped_column(
        "decision_reason",
        String(1000),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        "created_at",
        server_default=func.current_timestamp(),
        nullable=False,
    )
    decided_at: Mapped[Optional[datetime]] = mapped_column(
        "decided_at",
        nullable=True,
    )

    channel = relationship("Channel", back_populates="status_events")
    requester = relationship("User", foreign_keys=[requested_by])
    approver  = relationship("User", foreign_keys=[decided_by])
# ============================================================
#                      DB ENGINE & SESSION
# ============================================================
engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
)

SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# ---------- DB Session ----------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        async with session.begin():
            yield session


# ============================================================
#                      SECURITY / JWT
# ============================================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

def _truncate_bcrypt(p: str) -> str:
    # bcrypt limit is 72 *bytes*; เข้าง่ายๆ ด้วย utf-8 แล้วตัด
    b = p.encode("utf-8")
    if len(b) > 72:
        b = b[:72]
    return b.decode("utf-8", errors="ignore")

# def verify_password(plain: str, hashed: str) -> bool:
#     plain = _truncate_bcrypt(plain)
#     return pwd_context.verify(plain, hashed)

def verify_password(plain: str, stored: str) -> bool:
    # ตอนนี้รหัสใน DB ยังเป็น plain text อยู่
    return plain == stored


def hash_password(plain: str) -> str:
    plain = _truncate_bcrypt(plain)
    return pwd_context.hash(plain)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)

async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    stmt = select(User).where(User.username == username)
    res = await db.execute(stmt)
    return res.scalar_one_or_none()

async def get_user_by_name(db: AsyncSession, name: str) -> Optional[User]:
    stmt = select(User).where(User.name == name)
    res = await db.execute(stmt)
    return res.scalar_one_or_none()

async def get_user_by_id(db: AsyncSession, uid: int) -> Optional[User]:
    stmt = select(User).where(User.users_id == uid)
    res = await db.execute(stmt)
    return res.scalar_one_or_none()

async def authenticate_user(db: AsyncSession, username: str, password: str) -> Optional[User]:
    user = await get_user_by_username(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        if sub is None:
            raise credentials_exception
        uid = int(sub)   # id ถูกเก็บเป็น string ใน token → แปลงกลับเป็น int
    except (JWTError, ValueError):
        raise credentials_exception

    user = await get_user_by_id(db, uid)
    if not user:
        raise credentials_exception
    return user

async def get_owned_session(
    db: AsyncSession,
    session_id: int,
    user_id: int,
):
    stmt = (
        select(sessions)
        .where(
            sessions.sessions_id == session_id,
            sessions.user_id == user_id,   
        )
    )
    res = await db.execute(stmt)
    return res.scalar_one_or_none()


# ============================================================
#                      RAG / AI HELPERS
# ============================================================

async def call_ai(messages: List[dict], channel_id: int,session_id: int ) -> str:

    last_user_msg = None
    for m in reversed(messages):
        if m["role"] == "user":
            last_user_msg = m["content"]
            break
    if last_user_msg is None:
        last_user_msg = "สรุปข้อมูลจากฐานเอกสารให้หน่อย"

    loop = asyncio.get_running_loop()
    answer = await loop.run_in_executor(None, rag_engine.query, last_user_msg, channel_id, session_id)
    return answer

async def get_latest_pending_event( db: AsyncSession, channel_id: int) -> Optional[ChannelStatusEvent]:
    stmt = (
        select(ChannelStatusEvent)
        .where(ChannelStatusEvent.channel_id == channel_id,
               ChannelStatusEvent.decision.is_(None),
               ChannelStatusEvent.new_status == RoleChannel.public
               )
        .order_by(desc(ChannelStatusEvent.created_at))
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()

# ============================================================
#                      Pydantic SCHEMAS
# ============================================================
class UserCreate(BaseModel):
    username: str
    name: str
    password: str
    email: Optional[EmailStr] = None   # ตารางอนุญาตให้เป็น NULL

class UserOut(BaseModel):
    users_id: int
    username: str
    name: str
    email: Optional[EmailStr] = None
    role: RoleUser
    created_at: datetime
    class Config:
        from_attributes = True
        
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

class ChannelCreate(BaseModel):
    title: str
    description: Optional[str] = None

class ChannelResponse(BaseModel):
    channels_id: str
    title: str
    description: Optional[str] = None
    created_by: int
    status: RoleChannel
    
    @field_validator('channels_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int):
            return encode_id(v)
        return v

class ChannelOut(BaseModel):
    channels_id: str
    title: str
    description: Optional[str]
    status: RoleChannel
    created_at: datetime
    files: List[dict] 
    
    @field_validator('channels_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int):
            return encode_id(v)
        return v

class ChannelUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    # status: Optional[RoleChannel] = None

class ChannelUpdateResponse(BaseModel):
    title: str
    description: Optional[str] = None
    
class ChannelOneResponse(BaseModel):
    channels_id: str
    title: str
    description: Optional[str] = None
    status: RoleChannel
    created_by_id: int
    created_by_name: str
    created_at: datetime
    file_count: int
    files: List[dict]
    
    @field_validator('channels_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int):
            return encode_id(v)
        return v

class ChannelListPendingItem(BaseModel):
    channels_id: str
    title: str
    description: Optional[str] = None
    status: RoleChannel
    created_by_id: int
    created_by_name: str
    created_at: datetime
    file_count: int
    files: List[dict]
    
    @field_validator('channels_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int):
            return encode_id(v)
        return v

class ChannelListPublicItem(BaseModel):
    channels_id: str
    title: str
    description: Optional[str] = None
    status: RoleChannel
    created_by_id: int
    created_by_name: str
    created_at: datetime
    file_count: int
    files: List[dict] 
    
    @field_validator('channels_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int):
            return encode_id(v)
        return v

class ChannelListAllItem(BaseModel):
    channels_id: str
    title: str
    description: Optional[str] = None
    status: RoleChannel
    created_by_id: int
    created_by_name: str
    created_at: datetime
    file_count: int
    files: List[dict] 

    @field_validator('channels_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int):
            return encode_id(v)
        return v

class ChannelUpdateStatus(BaseModel):
    channels_id: str
    status: RoleChannel

class sessionCreate(BaseModel):
    channel_id: str
    
    @field_validator('channel_id', mode='before')
    def decode_channel_id(cls, v):
        if isinstance(v, str):
            decoded = decode_id(v)
            if decoded is None:
                raise ValueError("Invalid Channel ID")
            return decoded
        return v

class chatCreate(BaseModel):
    channels_id: str
    users_id: int
    sessions_id: int
    message: str
    sender_type: RoleSender
    
    @field_validator('channels_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int):
            return encode_id(v)
        return v

class chatHistoryItem(BaseModel):
    chat_id: int
    channels_id: str
    users_id: int
    sessions_id: int
    message: str
    sender_type: RoleSender
    created_at: datetime
    
    @field_validator('channels_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int):
            return encode_id(v)
        return v

class ChatRequest(BaseModel):
    sessions_id: str
    message: str
    
    @field_validator('sessions_id', mode='before')
    def encode_session_id(cls, v):
        if isinstance(v, str):
            real_id = decode_id(v)
            if real_id is None:
                raise ValueError("Invalid Session ID")
            return real_id
        return v 

class ModerationResponse(BaseModel):
    channels_id: str
    old_status: RoleChannel
    current_status: RoleChannel
    event_id: int
    message: str
    
    @field_validator('channels_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int):
            return encode_id(v)
        return v
    
class AdminDecisionIn(BaseModel):
    approve: bool
    reason: Optional[str] = None

class AdminDecisionOut(BaseModel):
    channels_id: str
    decision: ModerationDecision
    status_after: RoleChannel
    event_id: int
    decided_by: int
    decided_at: datetime
    message: str

    @field_validator('channels_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int):
            return encode_id(v)
        return v
    
class SessionResponse(BaseModel):
    sessions_id: str  = Field(..., validation_alias="sessions_id")
    channel_id: str
    user_id: int
    created_at: datetime
    
    @field_validator('sessions_id', mode='before')
    def encode_session_id(cls, v):
        if isinstance(v, int):
            return encode_id(v) 
        return v    

    @field_validator('channel_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int):
            return encode_id(v)
        return v
class FileDetail(BaseModel):
    files_id: str = Field(..., description="Hashed ID ของไฟล์")
    original_filename: str = Field(..., description="ชื่อไฟล์เดิม")
    size_bytes: int = Field(..., description="ขนาดไฟล์ (bytes)")
    mime: str = Field(..., description="ชนิดไฟล์ (MIME type)")
    channel_id: int = Field(..., description="ID ของ Channel ที่อัปโหลด")
    public_url: Optional[str] = Field(None, description="URL สำหรับเข้าถึงไฟล์ (มีเฉพาะกรณี public channel)")
    
    @field_validator('channel_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int):
            return encode_id(v)
        return v

class FileUploadResponse(BaseModel):
    files: list[FileDetail]

class FileListItem(BaseModel):
    files: list[FileDetail]

class UserRequestChannelStatusEventResponse(BaseModel):
    event_id: int
    channel_id: str
    old_status: RoleChannel
    new_status: RoleChannel
    requested_by: int
    
    # เพิ่ม field เหล่านี้ให้ครบตามที่คุณพยายาม assign ด้านล่าง
    # ควรใส่เป็น Optional เพราะ event ที่เพิ่งสร้างอาจยังไม่มีผลการตัดสิน
    decided_by: Optional[int] = None 
    decision: Optional[str] = None   # หรือเป็น Enum ถ้ามี
    decision_reason: Optional[str] = None
    decided_at: Optional[datetime] = None
    
    created_at: datetime

    @field_validator('channel_id', mode='before')
    def encode_channel_id(cls, v):
        if isinstance(v, int):
            return encode_id(v) # สมมติว่ามีฟังก์ชันนี้
        return v

# ============================================================
#                  APP INITIALIZATION / MIDDLEWARE
# ============================================================

app = FastAPI(title="FastAPI + MariaDB + JWT")
templates = Jinja2Templates(directory="templates")

# ---------- File/Static ----------
UPLOAD_ROOT = settings.upload_root
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app.mount("/static/uploads", StaticFiles(directory=UPLOAD_ROOT), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://192.168.1.122:3000", 
                   "http://localhost:3000", 
                   "http://127.0.0.1:3000",
                   "http://127.0.0.1:5500",
                   "https://lukeenortaed.site", 
                   "https://www.lukeenortaed.site",
                   "https://*.ngrok-free.app"],
    allow_origin_regex=r"https://.*\.ngrok-free\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# router = fastapi.APIRouter()
# fsd.install(router)
# app.include_router(router)


#     # สร้างตารางอัตโนมัติ (เหมาะกับ dev/POC) — โปรดใช้ Alembic ในงานจริง
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)


# ============================================================
#                  BASIC / HEALTH ROUTES
# ============================================================
@app.get("/",status_code=200,response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/healthz", response_class=PlainTextResponse)
def healthz_get():
    return "ok"

@app.head("/healthz")
def healthz_head():
    return Response(status_code=200)

base64_icon = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    image_bytes = base64.b64decode(base64_icon)
    
    return Response(content=image_bytes, media_type="image/png")

@app.head("/")
def root_head():
    return Response(status_code=200)

@app.get("/hashids-demo", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("hashids_demo.html", {
        "request": request,
        "salt_preview": settings.HASH_SALT[:3] + "***"
    })

# 3. POST Route: รวม Logic ทั้ง Encode และ Decode
@app.post("/hashids-demo", response_class=HTMLResponse)
async def process_hashids(
    request: Request,
    action: str = Form(...),              # ตัวบอกว่ากดปุ่มไหน (encode หรือ decode)
    number_input: Optional[int] = Form(None), # รับค่าตัวเลข (ถ้ามี)
    hash_input: Optional[str] = Form(None)    # รับค่า Hash (ถ้ามี)
):
    result_encode = None
    result_decode = None
    error_msg = None

    # Logic: ตรวจสอบว่า action คืออะไร
    if action == "encode" and number_input is not None:
        result_encode = hasher.encode(number_input)
        
    elif action == "decode" and hash_input:
        decoded = hasher.decode(hash_input)
        if decoded:
            result_decode = decoded[0]
        else:
            error_msg = "ไม่สามารถถอดรหัสได้ (Invalid Hash)"

    # ส่งค่ากลับไปที่ template เดิม
    return templates.TemplateResponse("hashids_demo.html", {
        "request": request,
        "salt_preview": settings.HASH_SALT[:3] + "***",
        "action": action,             # ส่งกลับไปเพื่อเช็คว่าเพิ่งทำอะไรเสร็จ
        "number_input": number_input, # ส่งค่าเดิมกลับไปแสดงในช่อง
        "hash_input": hash_input,     # ส่งค่าเดิมกลับไปแสดงในช่อง
        "result_encode": result_encode,
        "result_decode": result_decode,
        "error_msg": error_msg
    })

# ============================================================
#                  AUTH ROUTES
# ============================================================
# ออก access token ด้วย username/password จาก DB
@app.post("/auth/token")
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    user = await authenticate_user(db, form.username, form.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    # sub ต้องเป็น string ตามข้อแนะนำของ JWT
    access_token = create_access_token(data={"sub": str(user.users_id)})
    return {"access_token": access_token, "token_type": "bearer"}


# ============================================================
#                  USER ROUTES (CRUD + ROLE)
# ============================================================
# --- CRUD User ---
# add user
@app.post("/users", response_model=UserOut, status_code=201)
async def register_user(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    user = User(
        username=payload.username,
        name=payload.name,
        hashed_password=payload.password,
        email=payload.email,        
    )
    db.add(user)
    try:
        await db.flush()
    except IntegrityError:
        raise HTTPException(status_code=409, detail="Name or email already exists")

    # โหลดค่าที่ DB เติมให้ (เช่น id/created_at/role default)
    await db.refresh(user)
    return user

# Read: Get user by id
@app.get("/users/{user_id}", response_model=UserOut)
async def get_user_by_id_api(
    user_id: int = Path(..., gt=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # เฉพาะ admin หรือเจ้าของเท่านั้นที่ดูได้
    if current_user.role != RoleUser.admin and current_user.users_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    return user

# Update: Update user info (username, name, email)
@app.put("/users/{user_id}", response_model=UserOut)
async def update_user(
    user_id: int,
    payload: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if current_user.role != RoleUser.admin and current_user.users_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    if payload.username is not None:
        user.username = payload.username
    if payload.name is not None:
        user.name = payload.name
    if payload.email is not None:
        user.email = payload.email
    try:
        await db.flush()
    except IntegrityError:
        raise HTTPException(status_code=409, detail="Username, name, or email already exists")
    await db.refresh(user)
    return user

@app.put("/users/password/{user_id}", status_code=204)
async def update_user_password(
    user_id: int = Path(..., gt=0),
    payload: UserPasswordUpdate = Body(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if payload.new_password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="New password and confirmation do not match")
    if current_user.role != RoleUser.admin and current_user.users_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    if not verify_password(payload.old_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Old password is incorrect")
    user.hashed_password = payload.new_password
    await db.flush()
    return

# Delete: Delete user
@app.delete("/users/{user_id}", status_code=204)
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if current_user.role != RoleUser.admin and current_user.users_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    await db.delete(user)
    return

@app.put("/user/role/{user_id}/{new_role}", response_model=UserOut)
async def update_user_role(
    user_id: int = Path(..., gt=0),
    new_role: RoleUser = Path(...),
    secret_key: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="Not authorized")

    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if secret_key != "4+2*3=24":
        raise HTTPException(status_code=403, detail="secret_key is incorrect")

    user.role = new_role
    await db.flush()
    await db.refresh(user)

    return user

# Read: List all users (admin only)
@app.get("/users/list/", response_model=List[UserOut])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    stmt = select(User).offset(skip).limit(limit)
    result = await db.execute(stmt)
    users = result.scalars().all()
    return users

# Protected endpoint
@app.get("/get/userinfo/bytoken")
async def get_user_by_token(current_user: User = Depends(get_current_user)):
    return {
        "users_id": current_user.users_id,
        "username": current_user.username,
        "name": current_user.name,
        "email": current_user.email,
        "role": current_user.role,
    }


# ============================================================
#                  CHANNEL & FILE SETTINGS
# ============================================================
# --- CRUD Channel & File ---

# ตรวจไฟล์ (optional)
MAX_SIZE_PER_FILE = 50 * 1024 * 1024  # 50 MB
ALLOW_MIME = {"application/pdf",
              "text/plain"}

try:
    import magic
    def sniff_mime(path: pathlib.Path) -> str:
        return magic.from_file(str(path), mime=True) or "application/octet-stream"
except Exception:
    def sniff_mime(path: pathlib.Path) -> str:
        # fallback อย่างน้อยใช้สกุลไฟล์
        return "application/pdf" if path.suffix.lower() == ".pdf" else "application/octet-stream"

def _build_storage_path(channel_id: int, filename: str) -> tuple[pathlib.Path, str]:
    """คืน (abs_path, relative_path) โดย relative ใช้เก็บใน DB"""
    ext = pathlib.Path(filename or "").suffix.lower()
    uid = secrets.token_hex(16)  # uuid ก็ได้
    rel = pathlib.Path(str(channel_id)) / f"{uid}{ext}"
    abs_path = UPLOAD_ROOT / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    return abs_path, str(rel).replace("\\", "/")  # normalize

async def _save_upload_atomic(uf: UploadFile, final_path: pathlib.Path, max_size: int) -> int:
    """เขียนไฟล์ลง temp แล้ว atomic replace ไปยังปลายทาง"""
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    size_bytes = 0
    async with aiofiles.open(tmp_path, "wb") as f:
        while True:
            chunk = await uf.read(1024 * 1024)
            if not chunk:
                break
            size_bytes += len(chunk)
            if size_bytes > max_size:
                await uf.close()
                raise HTTPException(status_code=413, detail=f"File too large: {uf.filename}")
            await f.write(chunk)
    # replace แบบ atomic
    os.replace(tmp_path, final_path)
    return size_bytes


# ============================================================
#                  CHANNEL ROUTES
# ============================================================
@app.post("/channels", status_code=201, response_model=ChannelResponse)
async def create_channel(
    channel_in: ChannelCreate,  
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    new_channel = Channel(
        title=channel_in.title,
        description=channel_in.description,
        created_by=current_user.users_id,
        status=RoleChannel.private
    )

    db.add(new_channel)
    await db.flush()
    await db.refresh(new_channel) 
    return new_channel



@app.get("/channels/{channel_id}", response_model=ChannelOut)
async def get_channel_details(
    channel_id: str, 
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
    ):
    
    decoded_channel_id = decode_id(channel_id)

    result = await db.execute(select(Channel).where(Channel.channels_id == decoded_channel_id))
    channel = result.scalar_one_or_none()
    if channel is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    is_private_like = channel.status in (RoleChannel.private , RoleChannel.pending)
    is_owner = channel.created_by == current_user.users_id
    is_admin = current_user.role == RoleUser.admin
        
    # ตรวจสอบสิทธิ์การเข้าถึง
    if is_private_like and not (is_owner or is_admin):
        raise HTTPException(status_code=403, detail="Not authorized to access this channel")
    
    # ดึงรายการไฟล์ที่อยู่ใน channel นี้
    result = await db.execute(select(File).where(File.channel_id == channel_id))
    files = result.scalars().all()
    
    file_list = []
    for f in files:
        item = {
            "files_id": f.files_id,
            "original_filename": f.original_filename,
            "size_bytes": f.size_bytes,
            "created_at": f.created_at,
        }
        if channel.status == RoleChannel.public:
            item["public_url"] = f"/static/uploads/{f.storage_uri}"
        file_list.append(item)
    
    return {
        "channels_id": channel.channels_id,
        "title": channel.title,
        "description": channel.description,
        "status": channel.status,
        "created_at": channel.created_at,
        "files": file_list,
    }


@app.delete("/channels/{channel_id}")
async def delete_channel(channel_id: str, db: AsyncSession = Depends(get_db),current_user: User = Depends(get_current_user)):
    decoded_channel_id = decode_id(channel_id)
    # ดึง channel ตาม id
    result = await db.execute(
        select(Channel).where(Channel.channels_id == decoded_channel_id)
    )
    channel = result.scalar_one_or_none()

    if channel is None:
        # ถ้าไม่เจอ ให้คืน 404
        raise HTTPException(status_code=404, detail="Channel not found")

    if current_user.role != RoleUser.admin and channel.created_by != current_user.users_id:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to delete this channel"
        ) 
    
    flie_row_result = await db.execute(select(File).where(File.channel_id == channel.channels_id))
    flie_row = flie_row_result.scalars().all()
    
    # ลบเอกสารใน RAG (Chroma) ทั้งหมดที่เกี่ยวข้องกับไฟล์ใน channel นี้
    for file in flie_row:            
        try:
            # ลบเอกสารใน RAG (Chroma)
            rag_engine.delete_documents_by_file_id(file.files_id)
        except Exception as e:
            print(f"[RAG] failed to delete documents for file_id {file.files_id}: {e}")
    
    # ลบไฟล์ในช่องทั้ง DB + ดิสก์
    for fr in flie_row:
        try:
            (UPLOAD_ROOT / fr.storage_uri).unlink(missing_ok=True)
        except Exception:
            pass
        await db.delete(fr)
    await db.delete(channel)
    return {"message": "Channel deleted successfully"}

@app.put("/channels/{channel_id}", response_model=ChannelUpdateResponse)
async def update_channel(
    channel_id: str,
    payload: ChannelUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    decoded_channel_id = decode_id(channel_id)
    stmt = select(Channel).where(Channel.channels_id == decoded_channel_id)
    result = await db.execute(stmt)
    
    channels = result.scalars().one_or_none() 

    if channels is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    isNotOwner = channels.created_by != current_user.users_id
    isNotAdmin = current_user.role != RoleUser.admin
    
    if isNotOwner and isNotAdmin:
        raise HTTPException(status_code=403, detail="Only owner or admin can update channel")

    channels.title = payload.title
    channels.description = payload.description

    await db.flush() 
    
    await db.refresh(channels)
    
    return channels

@app.post("/channels/{channel_id}/request-public", response_model=ModerationResponse, status_code=201)
async def request_make_public(
    channel_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1) โหลด channel
    decoded_channel_id = decode_id(channel_id)
    res = await db.execute(select(Channel).where(Channel.channels_id == decoded_channel_id))
    channel = res.scalar_one_or_none()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    # 2) ต้องเป็นเจ้าของเท่านั้น
    if channel.created_by != current_user.users_id:
        raise HTTPException(status_code=403, detail="Only owner can request")

    # 3) ยื่นคำขอได้เฉพาะเมื่อสถานะเป็น private เท่านั้น
    if channel.status == RoleChannel.public:
        raise HTTPException(status_code=400, detail="Channel already public")
    if channel.status == RoleChannel.pending:
        raise HTTPException(status_code=409, detail="Channel already pending approval")

    old_status = channel.status
    channel.status = RoleChannel.pending

    # 4) สร้าง event
    event = ChannelStatusEvent(
        channel_id=channel.channels_id,
        old_status=old_status,
        new_status=RoleChannel.public,
        requested_by=current_user.users_id,
        decision=None,
    )
    db.add(event)
    await db.flush()
    await db.refresh(channel)
    await db.refresh(event)

    return ModerationResponse(
        channels_id=channel.channels_id,
        old_status=old_status,
        current_status=channel.status,
        event_id=event.event_id,
        message="Request submitted. Waiting for admin approval."
    )

@app.post("/channels/{channel_id}/cancel-request", response_model=ModerationResponse, status_code=201)
async def cancel_request_make_public(
    channel_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1) Decode และตรวจสอบ ID
    decoded_channel_id = decode_id(channel_id)
    if decoded_channel_id is None:
        raise HTTPException(status_code=404, detail="Invalid Channel ID")

    # 2) โหลด channel
    res = await db.execute(select(Channel).where(Channel.channels_id == decoded_channel_id))
    channel = res.scalar_one_or_none()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    # 3) ต้องเป็นเจ้าของเท่านั้น
    if channel.created_by != current_user.users_id:
        raise HTTPException(status_code=403, detail="Only owner can request")

    # 4) ยกเลิกได้เฉพาะเมื่อสถานะเป็น pending เท่านั้น
    if channel.status == RoleChannel.public:
        raise HTTPException(status_code=400, detail="Channel is already public")
    if channel.status == RoleChannel.private:
        raise HTTPException(status_code=409, detail="Channel is already private (no request to cancel)")

    # --- เก็บสถานะเดิม (Pending) ---
    old_status = channel.status
    
    # --- [FIXED] เปลี่ยนสถานะกลับเป็น Private ---
    channel.status = RoleChannel.private

    # 5) สร้าง event (Auto-resolved)
    # เราบันทึกว่ามีการเปลี่ยนกลับเป็น private แต่ใส่ข้อมูลว่า approved แล้ว (โดยตัว user เอง)
    # เพื่อไม่ให้ไปค้างในรายการรออนุมัติของ Admin
    now = datetime.now(timezone.utc)
    event = ChannelStatusEvent(
        channel_id=channel.channels_id,
        old_status=old_status,
        new_status=RoleChannel.private,
        requested_by=current_user.users_id,
        
        # [IMPROVED] บันทึกว่าตัดสินใจแล้วทันที ไม่ต้องรอ Admin
        decided_by=current_user.users_id, 
        decision=ModerationDecision.approved, # อนุมัติให้กลับเป็น private
        decision_reason="User cancelled the public request",
        decided_at=now
    )
    
    db.add(event)
    await db.flush()
    await db.refresh(channel)
    await db.refresh(event)

    return ModerationResponse(
        channels_id=channel.channels_id,
        old_status=old_status,
        current_status=channel.status,
        event_id=event.event_id,
        message="Request cancelled. Channel reverted to private."
    )

@app.post("/channels/{channel_id}/moderate-public", response_model=AdminDecisionOut)
async def approved_rejected_public_request(
    channel_id: str,
    payload: AdminDecisionIn,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="Admin only")
    
    decoded_channel_id = decode_id(channel_id)
    res = await db.execute(select(Channel).where(Channel.channels_id == decoded_channel_id))
    channel = res.scalar_one_or_none()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    if channel.status != RoleChannel.pending:
        raise HTTPException(status_code=400, detail="Channel is not pending")

    event_table = await get_latest_pending_event(db, channel_id)
    if not event_table:
        raise HTTPException(status_code=404, detail="No pending request event found")
    now =  datetime.now(timezone.utc)
    
    reason = payload.reason
    if payload.approve:
        final_message = reason or "Approved — channel is now PUBLIC."
        channel.status = RoleChannel.public
        event_table.decision = ModerationDecision.approved
    else:
        final_message = reason or "Rejected — channel remains PRIVATE."
        channel.status = RoleChannel.private
        event_table.decision = ModerationDecision.rejected

    event_table.decided_by = current_user.users_id
    event_table.decided_at = now
    event_table.decision_reason = reason

    await db.flush()
    await db.refresh(channel)
    await db.refresh(event_table)

    return AdminDecisionOut(
        channels_id=channel.channels_id,
        decision=event_table.decision,
        status_after=channel.status,
        event_id=event_table.event_id,
        decided_by=event_table.decided_by,
        decided_at=event_table.decided_at,
        message=final_message,
    )

@app.post("/channels/{channel_id}/admin-forced-public", response_model=AdminDecisionOut)
async def admin_forced_this_channel_to_be_public(
    channel_id: str,
    payload: AdminDecisionIn,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1. Check Admin Permission
    if current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="Admin only")
    
    # 2. Get Channel
    real_channel_id = decode_id(channel_id)
    if real_channel_id is None:
        raise HTTPException(status_code=404, detail="Invalid Channel ID")
    res = await db.execute(select(Channel).where(Channel.channels_id == real_channel_id))
    channel = res.scalar_one_or_none()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    # 3. Validation: ทำได้เฉพาะช่องที่เป็น Public หรือ Pending เท่านั้น
    if channel.status != RoleChannel.private and channel.status != RoleChannel.pending:
        raise HTTPException(status_code=400, detail="Channel is already private")

    # เก็บสถานะเดิมไว้ทำ Log
    old_status = channel.status
    now = datetime.now(timezone.utc)
    reason = payload.reason or "Admin forced channel to PUBLIC."

    # 4. Action: เปลี่ยนสถานะเป็น Public
    channel.status = RoleChannel.public
    
    # 5. Create NEW Event Log
    # เราสร้าง Event ใหม่เลย เพราะนี่คือการกระทำใหม่จาก Admin (ไม่ใช่การอนุมัติคำขอเก่า)
    event = ChannelStatusEvent(
        channel_id=channel.channels_id,
        old_status=old_status,
        new_status=RoleChannel.public,
        requested_by=current_user.users_id, # Admin เป็นคนเริ่ม
        decided_by=current_user.users_id,   # Admin เป็นคนตัดสิน
        decision=ModerationDecision.approved,
        decision_reason=reason,
        created_at=now,
        decided_at=now
    )

    db.add(event)
    await db.flush()
    await db.refresh(channel)
    await db.refresh(event)

    return AdminDecisionOut(
        channels_id=channel.channels_id,
        decision=event.decision,
        status_after=channel.status,
        event_id=event.event_id,
        decided_by=event.decided_by,
        decided_at=event.decided_at,
        message=reason,
    )

@app.post("/channels/{channel_id}/admin-forced-private", response_model=AdminDecisionOut)
async def admin_forced_this_channel_to_be_private(
    channel_id: str,
    payload: AdminDecisionIn,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    real_channel_id = decode_id(channel_id)
    if real_channel_id is None:
        raise HTTPException(status_code=404, detail="Invalid Channel ID")
    
    # 1. Check Admin Permission
    if current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="Admin only")
    
    # 2. Get Channel
    res = await db.execute(select(Channel).where(Channel.channels_id == real_channel_id))
    channel = res.scalar_one_or_none()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    # 3. Validation: ทำได้เฉพาะช่องที่เป็น Public หรือ Pending เท่านั้น
    if channel.status != RoleChannel.public and channel.status != RoleChannel.pending:
        raise HTTPException(status_code=400, detail="Channel is already private")

    # เก็บสถานะเดิมไว้ทำ Log
    old_status = channel.status
    now = datetime.now(timezone.utc)
    reason = payload.reason or "Admin forced channel to PRIVATE."

    # 4. Action: เปลี่ยนสถานะเป็น Private
    channel.status = RoleChannel.private
    
    # 5. Create NEW Event Log
    # เราสร้าง Event ใหม่เลย เพราะนี่คือการกระทำใหม่จาก Admin (ไม่ใช่การอนุมัติคำขอเก่า)
    event = ChannelStatusEvent(
        channel_id=channel.channels_id,
        old_status=old_status,
        new_status=RoleChannel.private,
        requested_by=current_user.users_id, # Admin เป็นคนเริ่ม
        decided_by=current_user.users_id,   # Admin เป็นคนตัดสิน
        decision=ModerationDecision.approved,
        decision_reason=reason,
        created_at=now,
        decided_at=now
    )

    db.add(event)
    await db.flush()
    await db.refresh(channel)
    await db.refresh(event)

    return AdminDecisionOut(
        channels_id=channel.channels_id,
        decision=event.decision,
        status_after=channel.status,
        event_id=event.event_id,
        decided_by=event.decided_by,
        decided_at=event.decided_at,
        message=reason,
    )


@app.post("/channels/{channels_id}/owner-set-private", response_model=ModerationResponse)
async def owner_set_this_channel_private(
    channels_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1. ค้นหา Channel
    decoded_channel_id = decode_id(channels_id)
    res = await db.execute(select(Channel).where(Channel.channels_id == decoded_channel_id))
    channel = res.scalar_one_or_none()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    # 2. ตรวจสอบสิทธิ์: ต้องเป็นเจ้าของ Channel หรือ Admin เท่านั้น
    if channel.created_by != current_user.users_id and current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="Only owner can set channel to private")

    # 3. ตรวจสอบสถานะปัจจุบัน: ถ้าเป็น private อยู่แล้วไม่ต้องทำอะไร
    if channel.status == RoleChannel.private:
        raise HTTPException(status_code=400, detail="Channel is already private")

    old_status = channel.status
    channel.status = RoleChannel.private

    # 4. บันทึกประวัติการเปลี่ยนสถานะใน ChannelStatusEvent
    event = ChannelStatusEvent(
        channel_id=channel.channels_id,
        old_status=old_status,
        new_status=RoleChannel.private,
        requested_by=current_user.users_id,
        decided_by=current_user.users_id,
        decision=ModerationDecision.approved, # ถือว่าอนุมัติทันทีโดยเจ้าของ
        decision_reason="Changed to private by owner",
        decided_at=datetime.now(timezone.utc)
    )
    db.add(event)
    
    await db.flush()
    await db.refresh(channel)
    await db.refresh(event)

    return ModerationResponse(
        channels_id=channel.channels_id,
        old_status=old_status,
        current_status=channel.status,
        event_id=event.event_id,
        message="Channel is now private."
    )

# @app.put("/channels/admin/debug/force-status", response_model=ChannelUpdateStatus)
# async def update_channel_status(
#     channel_id: int,
#     new_status: RoleChannel = Body(..., embed=True),
#     db: AsyncSession = Depends(get_db),
#     current_user: User = Depends(get_current_user),
# ):
#     result = await db.execute(select(Channel).where(Channel.channels_id == channel_id))
#     channel = result.scalar_one_or_none()
#     if channel is None:
#         raise HTTPException(status_code=404, detail="Channel not found")
#     # เฉพาะ admin เท่านั้นที่แก้ไขได้
#     if current_user.role != RoleUser.admin:
#         raise HTTPException(status_code=403, detail="Admin only")
    
#     channel.status = new_status
#     await db.flush()
#     await db.refresh(channel)
    
#     return ChannelUpdateStatus(
#         channels_id=channel.channels_id,
#         status=channel.status,
#     )

@app.get("/channels/pending/list/", response_model=List[ChannelListPendingItem])
async def list_pending_channels(
    search_by_name: str | None = Query(None, description="ค้นหาจากชื่อ"),
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Channel)
        .options(
            joinedload(Channel.creator), 
            selectinload(Channel.files) 
        )
        .where(Channel.status == RoleChannel.pending)
        .order_by(Channel.created_at.desc())
    )

    if search_by_name:
        stmt = stmt.where(Channel.title.like(f"%{search_by_name}%"))

    stmt = stmt.offset(skip).limit(limit)

    result = await db.execute(stmt)
    channels = result.scalars().all()

    channel_list = []
    for ch in channels:
        
        file_list = [
            {
                "files_id": f.files_id,
                "original_filename": f.original_filename,
                "storage_uri": f.storage_uri,
                "size_bytes": f.size_bytes,
                "created_at": f.created_at,
            }
            for f in ch.files #
        ]

        channel_list.append(ChannelListPendingItem(
            channels_id=ch.channels_id,
            title=ch.title,
            description=ch.description,
            status=ch.status,
            created_by_id=ch.created_by,
            created_by_name=ch.creator.username if ch.creator else "Unknown", # เข้าถึงผ่าน relationship
            created_at=ch.created_at,
            file_count=len(ch.files),
            files=file_list,
        ))
        
    return channel_list


@app.get("/channels/public/list/", response_model=List[ChannelListPublicItem])
async def list_public_channels(
    search_by_name: str | None = Query(None, description="ค้นหาจากชื่อ"),
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    # 1. สร้าง Base Query พร้อม Eager Load (ดึง User และ Files ทีเดียว)
    stmt = (
        select(Channel)
        .options(
            joinedload(Channel.creator), # ดึงข้อมูลคนสร้าง (User)
            selectinload(Channel.files)  # ดึงข้อมูลไฟล์ (Files)
        )
        .where(Channel.status == RoleChannel.public)
        .order_by(Channel.created_at.desc())
    )

    # 2. ใส่ Filter (Where) ก่อน Pagination
    if search_by_name:
        stmt = stmt.where(Channel.title.like(f"%{search_by_name}%"))

    # 3. ปิดท้ายด้วย Pagination (Offset/Limit)
    stmt = stmt.offset(skip).limit(limit)

    result = await db.execute(stmt)
    channels = result.scalars().all()

    # 4. แปลงข้อมูล (ไม่ต้อง Query เพิ่มแล้ว เพราะดึงมาหมดแล้วข้างบน)
    channel_list = []
    for ch in channels:
        
        # Map ไฟล์จาก memory ได้เลย
        file_list = [
            {
                "files_id": f.files_id,
                "original_filename": f.original_filename,
                "storage_uri": f.storage_uri,
                "size_bytes": f.size_bytes,
                "created_at": f.created_at,
                # ถ้า public ควรมี link ให้โหลดได้เลย
                "public_url": f"/static/uploads/{f.storage_uri}" 
            }
            for f in ch.files
        ]

        channel_list.append(ChannelListPublicItem(
            channels_id=ch.channels_id,
            title=ch.title,
            description=ch.description,
            status=ch.status,
            created_by_id=ch.created_by,
            created_by_name=ch.creator.username if ch.creator else "Unknown", # ดึงชื่อจาก Relation
            created_at=ch.created_at,
            file_count=len(ch.files),
            files=file_list,
        ))

    return channel_list

@app.get("/channels/list/", response_model=List[ChannelOneResponse])
async def list_my_channels(
    search_by_name: str | None = Query(None, description="ค้นหาจากชื่อ"),
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    
    stmt = (
        select(Channel)
        .options(
            joinedload(Channel.creator), 
            selectinload(Channel.files) 
        )
        .where(Channel.created_by == current_user.users_id)
        .order_by(Channel.created_at.desc())
    )

    if search_by_name:
        stmt = stmt.where(Channel.title.like(f"%{search_by_name}%"))

    stmt = stmt.offset(skip).limit(limit)

    result = await db.execute(stmt)
    channels = result.scalars().all()

    channel_list = []
    for ch in channels:
        
        file_list = [
            {
                "files_id": f.files_id,
                "original_filename": f.original_filename,
                "storage_uri": f.storage_uri,
                "size_bytes": f.size_bytes,
                "created_at": f.created_at,
            }
            for f in ch.files #
        ]

        channel_list.append(ChannelOneResponse(
            channels_id=ch.channels_id,
            title=ch.title,
            description=ch.description,
            status=ch.status,
            created_by_id=ch.created_by,
            created_by_name=ch.creator.username if ch.creator else "Unknown", # เข้าถึงผ่าน relationship
            created_at=ch.created_at,
            file_count=len(ch.files),
            files=file_list,
        ))
    return channel_list

@app.get("/channels/list/all/", response_model=List[ChannelListAllItem])
async def list_all_channels(
    search_by_name: str | None = Query(None, description="ค้นหาจากชื่อ"),
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="Admin only")

    stmt = (
        select(Channel)
        .options(
            joinedload(Channel.creator), 
            selectinload(Channel.files) 
        )
        .order_by(Channel.created_at.desc())
    )

    if search_by_name:
        stmt = stmt.where(Channel.title.like(f"%{search_by_name}%"))

    stmt = stmt.offset(skip).limit(limit)

    result = await db.execute(stmt)
    channels = result.scalars().all()

    channel_list = []
    for ch in channels:
        
        file_list = [
            {
                "files_id": f.files_id,
                "original_filename": f.original_filename,
                "storage_uri": f.storage_uri,
                "size_bytes": f.size_bytes,
                "created_at": f.created_at,
            }
            for f in ch.files #
        ]

        channel_list.append(ChannelListPendingItem(
            channels_id=ch.channels_id,
            title=ch.title,
            description=ch.description,
            status=ch.status,
            created_by_id=ch.created_by,
            created_by_name=ch.creator.username if ch.creator else "Unknown",
            created_at=ch.created_at,
            file_count=len(ch.files),
            files=file_list,
        ))
    return channel_list

# ============================================================
#                  FILE ROUTES
# ============================================================
@app.get("/files/list/{channel_id}", response_model=FileListItem)
async def list_files_in_channel(
    channel_id: str = Path(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    real_channel_id = decode_id(channel_id)
    if real_channel_id is None:
         raise HTTPException(status_code=404, detail="Invalid Channel ID")
     
    # 1) ตรวจสอบว่า channel มีอยู่จริง
    res = await db.execute(select(Channel).where(Channel.channels_id == real_channel_id))
    channel = res.scalar_one_or_none()
    if channel is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    is_private = channel.status in (RoleChannel.private , RoleChannel.pending)
    is_owner = (channel.created_by == current_user.users_id)
    is_admin = (current_user.role == RoleUser.admin)
        
    # 2) ตรวจสอบสิทธิ์การเข้าถึง
    if is_private and not (is_owner or is_admin):
        raise HTTPException(status_code=403, detail="Not authorized to access this channel")

    # 3) ดึงรายการไฟล์
    res = await db.execute(select(File).where(File.channel_id == channel_id))
    files = res.scalars().all()

    file_list: list[FileDetail] = []
    for f in files:
        hashed_file_id = encode_id(f.files_id)
        file_resp = FileDetail(
            files_id=hashed_file_id,
            original_filename=f.original_filename,
            size_bytes=f.size_bytes,
            mime=sniff_mime(UPLOAD_ROOT / f.storage_uri),
            channel_id=channel_id,
            public_url=f"/static/uploads/{f.storage_uri}" if channel.status == RoleChannel.public else None
        )
        file_list.append(file_resp)

    return FileListItem(files=file_list)

@app.post("/files/upload", status_code=201, response_model=FileUploadResponse)
async def upload_files_only(
    channel_id: str = Form(...),
    files: list[UploadFile] = FastAPIFile(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    real_channel_id = decode_id(channel_id)
    if real_channel_id is None:
        raise HTTPException(status_code=404, detail="Invalid Channel ID")
    
    # 1) ตรวจสอบว่า channel มีอยู่จริง
    res = await db.execute(select(Channel).where(Channel.channels_id == real_channel_id))
    channel = res.scalar_one_or_none()
    if channel is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    isOwner = (channel.created_by == current_user.users_id)
    
    # 2) ตรวจสิทธิ์เจ้าของ channel
    if not isOwner:
        raise HTTPException(status_code=403, detail="Not authorized to upload to this channel")

    # 3) จำกัดจำนวนไฟล์ต่อ request
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Too many files in one request")

    stored_files: list[FileDetail] = []
    created_paths: list[pathlib.Path] = []

    try:
        for uf in files:
            final_path, rel_path = _build_storage_path(channel_id, uf.filename)

            # 4) เขียนไฟล์ + ตรวจขนาด
            size_bytes = await _save_upload_atomic(uf, final_path, MAX_SIZE_PER_FILE)
            created_paths.append(final_path)
            await uf.close()

            # 5) ตรวจ MIME
            detected_mime = sniff_mime(final_path)
            if detected_mime not in ALLOW_MIME:
                final_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {detected_mime}")

            # 6) บันทึก DB
            frow = File(
                uploaded_by=current_user.users_id,
                channel_id=channel_id,
                original_filename=uf.filename or final_path.name,
                storage_uri=rel_path,
                size_bytes=size_bytes,
            )
            db.add(frow)
            await db.flush()  # ได้ files_id

            # 7) เติมเอกสารเข้า RAG (Chroma)
            abs_path = UPLOAD_ROOT / rel_path
            try:
                docs = SimpleDirectoryReader(input_files=[str(abs_path)]).load_data()
                for d in docs:
                    d.metadata = d.metadata or {}
                    d.metadata["channel_id"] = str(channel_id)
                    d.metadata["filename"] = frow.original_filename
                    d.metadata["files_id"] = str(frow.files_id)

                # เติมเอกสารลงคอลเลกชันเดิม
                rag_engine.add_documents(docs)

            except Exception as e:
                print(f"[RAG] failed to index {abs_path}: {e}")

            hashed_file_id = encode_id(frow.files_id)
            file_resp = FileDetail(
                files_id=hashed_file_id,
                original_filename=frow.original_filename,
                size_bytes=size_bytes,
                mime=detected_mime,
                channel_id=channel_id,
                public_url=f"/static/uploads/{rel_path}" if channel.status == RoleChannel.public else None
            )
            stored_files.append(file_resp)

    except Exception:
        for p in created_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        raise

    return FileUploadResponse(files=stored_files)


@app.delete("/files/delete/{file_id}", status_code=204)
async def delete_file(
    file_id: str = Path(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # แปลง hash กลับเป็น file_id
    file_id = decode_id(file_id)
    # 1. Query แบบ Join เพื่อลดการเรียก DB หลายรอบ
    stmt = (
        select(File)
        .options(joinedload(File.channel)) 
        .where(File.files_id == file_id)
    )
    result = await db.execute(stmt)
    file = result.scalar_one_or_none()

    if file is None:
        raise HTTPException(status_code=404, detail="File not found")

    # 2. Check Permission (Logic เดิมแต่เขียนให้กระชับ)
    is_admin = current_user.role == RoleUser.admin
    is_file_owner = file.uploaded_by == current_user.users_id
    is_channel_owner = file.channel and file.channel.created_by == current_user.users_id

    if not (is_admin or is_file_owner or is_channel_owner):
        raise HTTPException(status_code=403, detail="Not authorized")

    # 3. เตรียม Path (แก้ Hardcoded Path D:/...)
    # ควรใช้ Path สัมพัทธ์ หรือดึงจาก Env Config
    trash_folder = settings.TRASH_DIR
    src = UPLOAD_ROOT / file.storage_uri
    dst = trash_folder / f"{file.channel_id}_{file.files_id}_{src.name}"

    # 4. ย้ายไฟล์แบบ Non-blocking (ใช้ Thread แยก)
    # เพื่อไม่ให้ Server ค้างระหว่างย้ายไฟล์
    async def move_file_task():
        if src.exists():
            trash_folder.mkdir(parents=True, exist_ok=True)
            # shutil.move ปลอดภัยกว่า rename กรณีข้าม drive
            await asyncio.to_thread(shutil.move, str(src), str(dst))

    try:
        await move_file_task()
    except Exception as e:
        print(f"[FILE] Move to trash failed: {e}")
        # ตัดสินใจตรงนี้: ถ้าย้ายไฟล์ไม่ผ่าน จะให้ลบ DB ไหม? 
        # ปกติถ้าไฟล์จริงลบไม่ได้ ก็ไม่ควรลบ record ใน DB -> ควร raise Error
        raise HTTPException(status_code=500, detail="Failed to move file to trash")

    # 5. สั่งลบใน Session (รอ Auto-commit ทำงานตอนจบ request)
    await db.delete(file)
    
    # 6. ลบ RAG 
    # (ควรทำหลังจากมั่นใจว่าไฟล์ไปแล้ว)
    try:
        rag_engine.delete_documents_by_file_id(file.files_id)
    except Exception as e:
        print(f"[RAG] Warning: RAG deletion failed: {e}")

    return


# ============================================================
#                  SESSION ROUTES
# ============================================================
@app.post("/session", status_code=201 , response_model=SessionResponse)
async def create_session(
    payload: sessionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1) หา channel ก่อน
    result = await db.execute(
        select(Channel).where(Channel.channels_id == payload.channel_id)
    )
    channel = result.scalar_one_or_none()
    if channel is None:
        raise HTTPException(status_code=404, detail="Channel not found")

    # 2) เช็คสิทธิ์เข้า channel นี้
    # - public: ใครก็เข้าได้
    # - private: ต้องเป็นคนสร้าง หรือ admin
    if channel.status in (RoleChannel.private, RoleChannel.pending):
        if channel.created_by != current_user.users_id and current_user.role != RoleUser.admin:
            raise HTTPException(status_code=403, detail="Not authorized to access this channel")

    # 4) สร้าง session ใหม่
    new_session = sessions(
        channel_id=payload.channel_id,
        user_id=current_user.users_id,
    )
    db.add(new_session)
    await db.flush()
    await db.refresh(new_session)
    # rag_engine.debug_list_docs_by_channel(new_session.channel_id)
    return new_session

@app.delete("/session/delete/{session_id}", status_code=204)
async def delete_session(
    session_id: str = Path(..., title="The hashed session ID"), # 1. เปลี่ยนรับเป็น String (Hash)
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    real_session_id = decode_id(session_id)
    
    result = await db.execute(select(sessions).where(sessions.sessions_id == real_session_id))
    session = result.scalar_one_or_none()
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    isAdmin = (current_user.role == RoleUser.admin)
    isOwner = (session.user_id == current_user.users_id)
    
    if not isAdmin and not isOwner:
        raise HTTPException(status_code=403, detail="Not authorized to delete this session")
    
    await db.delete(session)
    
    rag_engine.clear_session_history(session.sessions_id) 
    
    return



# ============================================================
#                  CHAT + AI ROUTES
# ============================================================
@app.post("/sessions/ollama-reply", status_code=201)
async def Talking_with_Ollama_from_document(
    payload: ChatRequest = Body(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1) เช็คว่า session เป็นของ user นี้
    sess_stmt = (
        select(sessions)
        .where(
            sessions.sessions_id == payload.sessions_id,
            sessions.user_id == current_user.users_id,
        )
    )
    sess_res = await db.execute(sess_stmt)
    sess = sess_res.scalar_one_or_none()
    if sess is None:
        raise HTTPException(status_code=403, detail="Not your session")

    # 2) เซฟข้อความ user
    user_chat = chats(
        channels_id=sess.channel_id,
        users_id=current_user.users_id,
        sessions_id=sess.sessions_id,
        message=payload.message,
        sender_type=RoleSender.user,
    )
    db.add(user_chat)
    await db.flush()
    await db.refresh(user_chat)

    # ⚡️ 3) ส่งเฉพาะข้อความล่าสุดไปยัง AI
    ai_messages = [
        {"role": "user", "content": payload.message}
    ]

    # 4) เรียก RAG / AI
    ai_text = await call_ai(ai_messages, sess.channel_id, sess.sessions_id)

    # 5) เซฟคำตอบ AI
    ai_chat = chats(
        channels_id=sess.channel_id,
        users_id=current_user.users_id,
        sessions_id=sess.sessions_id,
        message=ai_text,
        sender_type=RoleSender.AI,
    )
    db.add(ai_chat)
    await db.flush()
    await db.refresh(ai_chat)

    # 6) ส่งกลับ
    return {
        "user_message": {
            "chat_id": user_chat.chat_id,
            "message": user_chat.message,
            "sender_type": user_chat.sender_type,
            "created_at": user_chat.created_at,
        },
        "ai_message": {
            "chat_id": ai_chat.chat_id,
            "message": ai_chat.message,
            "sender_type": ai_chat.sender_type,
            "created_at": ai_chat.created_at,
        },
    }




@app.get("/sessions/{session_id}/history", response_model=List[chatHistoryItem])
async def get_chat_history(session_id: str = Path(..., gt=0),
                           db: AsyncSession = Depends(get_db),
                           current_user: User = Depends(get_current_user)
                           ):
    real_session_id = decode_id(session_id)
    if real_session_id is None:
         raise HTTPException(status_code=404, detail="Invalid Session ID")
     
    # 1) ตรวจว่า session นี้เป็นของ user นี้จริงไหม
    owned_sess = await get_owned_session(db, real_session_id, current_user.users_id)
    if owned_sess is None:
        # ไม่ใช่ของเขา หรือไม่มีอยู่
        raise HTTPException(status_code=403, detail="Not your session")

    # 2) ดึงประวัติ chat ทั้งหมดใน session นี้
    result = await db.execute(
        select(chats)
        .where(chats.sessions_id == session_id)
        .order_by(chats.created_at.asc())
    )
    chat_rows = result.scalars().all()

    history = []
    for row in chat_rows:
        history.append(chatHistoryItem(
            chat_id=row.chat_id,
            channels_id=row.channels_id,
            users_id=row.users_id,
            sessions_id=row.sessions_id,
            message=row.message,
            sender_type=row.sender_type,
            created_at=row.created_at,
        ))

    return history

# ============================================================
#                 EVENT ROUTES
# ============================================================
@app.get("/events/request/{user_id}", response_model=List[UserRequestChannelStatusEventResponse])
async def get_channel_status_events_by_user(
    user_id: str = Path(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    real_user_id = decode_id(user_id)
    if real_user_id is None:
        raise HTTPException(status_code=404, detail="Invalid User ID")

    # 1) ตรวจสอบสิทธิ์: ต้องเป็น admin หรือ เจ้าของ user_id เท่านั้น
    if current_user.role != RoleUser.admin and current_user.users_id != real_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to view these events")

    # 2) ดึง events
    result = await db.execute(
        select(ChannelStatusEvent)
        .where(ChannelStatusEvent.requested_by == real_user_id)
        .order_by(ChannelStatusEvent.created_at.desc())
    )
    events = result.scalars().all()

    # 3) แปลงข้อมูลและส่งกลับ (Mapping)
    event_list = []
    for ev in events:
        # สร้าง Object response
        response_item = UserRequestChannelStatusEventResponse(
            event_id=ev.event_id,
            channel_id=ev.channel_id, # ตรงนี้ Validator จะทำงานแปลง int -> encoded str ให้
            old_status=ev.old_status,
            new_status=ev.new_status,
            requested_by=ev.requested_by,
            decided_by=ev.decided_by,
            decision=ev.decision,
            decision_reason=ev.decision_reason,
            created_at=ev.created_at,
            decided_at=ev.decided_at,
        )
        event_list.append(response_item)
    
    return event_list


# ============================================================
#                  DEBUG / UTIL ROUTES
# ============================================================
@app.post("/debug")
def debug_endpoint():
    rag_engine.debug_list_docs_by_channel(channel_id=6)
    return {"message": "Debug payload received"}
