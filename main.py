# main.py
import asyncio
from datetime import datetime, timedelta, timezone
import json
from typing import Optional, AsyncGenerator, List
# from urllib import response
import uuid, pathlib
import aiofiles
from fastapi import Body, FastAPI, APIRouter, Depends, File as FormFile, Form, UploadFile, HTTPException, status, Query, Path , Request, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# import httpx
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel , EmailStr
from pydantic_settings import BaseSettings
from sqlalchemy import String, desc, func, select ,Enum as SAEnum, ForeignKey, text
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
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

from rag_engine import add_documents, rag_query, rag_query_with_channel

# import fastapi
# import asyncio
# import fastapi_swagger_dark as fsd
# from fastapi_limiter import FastAPILimiter
# from fastapi_limiter.depends import RateLimiter
# from fastapi.responses import JSONResponse

API_KEY = os.getenv("API_KEY")
# ---------- Settings ----------
class Settings(BaseSettings):
    database_url: str
    secret_key: str = "dev-secret"
    access_token_expire_minutes: int = 720
    upload_root: pathlib.Path = pathlib.Path("./uploads")
    api_key: str = API_KEY
    class Config:
        env_file = ".env"

settings = Settings()



# ---------- DB Setup ----------
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
    
class ModerationDecision(str, enum.Enum):
    approved = "approved"
    rejected = "rejected"
    
class Channel(Base):
    __tablename__ = "channels"

    # PK: INT(10) UNSIGNED AUTO_INCREMENT
    channels_id: Mapped[int] = mapped_column(
        "channels_id",
        MyInt(unsigned=True),              # ตรงกับ UNSIGNED
        primary_key=True,
        autoincrement=True,
    )

    # title: NOT NULL
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
        SAEnum(RoleChannel),                       # ผูกกับ enum ของ MySQL
        nullable=False,
        default=RoleChannel.private,
        server_default=text("'private'"),      # ให้ default ฝั่ง DB ตรงกับสคีมา
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
        # ปล่อยให้ DB ใส่ค่าให้เองตาม default
        server_default=func.current_timestamp(),
        nullable=False,
    )
    
class User(Base):
    __tablename__ = "users"

    # PK: INT(10) UNSIGNED AUTO_INCREMENT
    users_id: Mapped[int] = mapped_column(
        "users_id",
        MyInt(unsigned=True),              # ตรงกับ UNSIGNED
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
        MyEnum(RoleUser),                       # ผูกกับ enum ของ MySQL
        nullable=False,
        server_default=text("'user'"),      # ให้ default ฝั่ง DB ตรงกับสคีมา
    )

    # created_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    created_at: Mapped[datetime] = mapped_column(
        "created_at",
        # ปล่อยให้ DB ใส่ค่าให้เองตาม default
        server_default=func.current_timestamp(),
        nullable=False,
    )

class File(Base):
    __tablename__ = "files"
    files_id: Mapped[int] = mapped_column("files_id", MyInt(unsigned=True), primary_key=True, autoincrement=True)
    uploaded_by: Mapped[Optional[int]] = mapped_column("uploaded_by", MyInt(unsigned=True), nullable=True)
    channel_id: Mapped[Optional[int]] = mapped_column("channel_id", MyInt(unsigned=True), nullable=True)
    original_filename: Mapped[str] = mapped_column("original_filename", String(512), nullable=False)
    storage_uri: Mapped[str] = mapped_column("storage_uri", String(1024), nullable=False)  # เก็บ path/URL ให้ชัดเจน
    size_bytes: Mapped[Optional[int]] = mapped_column("size_bytes", MyInt(unsigned=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column("created_at", server_default=func.current_timestamp(), nullable=False)

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

# ---------- Security / JWT ----------
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
            sessions.user_id == user_id,   # สำคัญ! ต้องเป็นของคนนี้จริง
        )
    )
    res = await db.execute(stmt)
    return res.scalar_one_or_none()


def get_rag_query_with_channel():
    from rag_engine import rag_query_with_channel
    return rag_query_with_channel


async def call_ai(messages: List[dict], channel_id: int) -> str:

    # หา user message ตัวท้ายสุด
    last_user_msg = None
    for m in reversed(messages):
        if m["role"] == "user":
            last_user_msg = m["content"]
            break
    if last_user_msg is None:
        last_user_msg = "สรุปข้อมูลจากฐานเอกสารให้หน่อย"

    # LlamaIndex เป็น sync → offload ไป thread จะปลอดภัยกว่า
    loop = asyncio.get_running_loop()
    # rag_query_with_channel_func = get_rag_query_with_channel()
    answer = await loop.run_in_executor(None, rag_query_with_channel, last_user_msg, channel_id)
    return answer

def get_rag_query():
    from rag_engine import rag_query
    return rag_query

def get_rag_index():
    from rag_engine import index
    return index    

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

# ---------- ตัวช่วยเซฟไฟล์แบบ async ----------
async def _save_upload_to_disk(uf: UploadFile, dst_path: pathlib.Path, max_size: int) -> int:
    """บันทึก UploadFile ลงดิสก์แบบ async และคืนค่า size (bytes)"""
    size_counter = 0
    async with aiofiles.open(dst_path, "wb") as f:
        while True:
            chunk = await uf.read(1024 * 1024)
            if not chunk:
                break
            size_counter += len(chunk)
            if size_counter > max_size:
                raise HTTPException(status_code=413, detail=f"File too large: {uf.filename}")
            await f.write(chunk)
    return size_counter

# ---------- Schemas ----------
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

class ChannelOut(BaseModel):
    channels_id: int
    title: str
    description: Optional[str]
    status: RoleChannel
    created_at: datetime
    files: List[dict]  # รายการไฟล์ใน channel นี้

class ChannelUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    # status: Optional[RoleChannel] = None

class ChannelListItem(BaseModel):
    channels_id: int
    title: str
    description: Optional[str] = None
    status: RoleChannel
    created_at: datetime
    file_count: int
    
class ChannelListPublicItem(BaseModel):
    channels_id: int
    title: str
    description: Optional[str] = None
    status: RoleChannel
    created_at: datetime
    files: List[dict]  # รายการไฟล์ใน channel นี้
    
class ChannelUpdateStatus(BaseModel):
    channels_id: int
    status: RoleChannel

class sessionCreate(BaseModel):
    channel_id: int

class chatCreate(BaseModel):
    channels_id: int
    users_id: int
    sessions_id: int
    message: str
    sender_type: RoleSender

class chatHistoryItem(BaseModel):
    chat_id: int
    channels_id: int
    users_id: int
    sessions_id: int
    message: str
    sender_type: RoleSender
    created_at: datetime

class message(BaseModel):
    message: str

class ModerationResponse(BaseModel):
    channel_id: int
    old_status: RoleChannel
    current_status: RoleChannel
    event_id: int
    message: str
    
class AdminDecisionIn(BaseModel):
    approve: bool
    reason: Optional[str] = None

class AdminDecisionOut(BaseModel):
    channels_id: int
    decision: ModerationDecision
    status_after: RoleChannel
    event_id: int
    decided_by: int
    decided_at: datetime
    message: str
# ---------- App ----------

app = FastAPI(title="FastAPI + MariaDB + JWT")
templates = Jinja2Templates(directory="templates")

# ---------- File/Static ----------
UPLOAD_ROOT = settings.upload_root
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app.mount("/static/uploads", StaticFiles(directory=UPLOAD_ROOT), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["192.168.1.122:3000", 
                   "http://localhost:3000", 
                   "http://127.0.0.1:3000",
                   "http://127.0.0.1:5500",
                   "https://lukeenortaed.site", 
                   "https://www.lukeenortaed.site"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# router = fastapi.APIRouter()
# fsd.install(router)
# app.include_router(router)

@app.on_event("startup")
async def on_startup():
    pass
#     # สร้างตารางอัตโนมัติ (เหมาะกับ dev/POC) — โปรดใช้ Alembic ในงานจริง
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)

@app.get("/",status_code=200,response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/healthz", response_class=PlainTextResponse)
def healthz_get():
    return "ok"

@app.head("/healthz")
def healthz_head():
    return Response(status_code=200)

# ถ้ายังอยากให้ HEAD ที่ "/" ใช้ได้ด้วย
@app.head("/")
def root_head():
    return Response(status_code=200)

# ออก access token ด้วย username/password จาก DB
@app.post("/auth/token")
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    user = await authenticate_user(db, form.username, form.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    # sub ต้องเป็น string ตามข้อแนะนำของ JWT
    access_token = create_access_token(data={"sub": str(user.users_id)})
    return {"access_token": access_token, "token_type": "bearer"}

# --- CRUD User ---
# add user
@app.post("/users", response_model=UserOut, status_code=201)
async def register_user(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    user = User(
        username=payload.username,
        name=payload.name,
        hashed_password=payload.password,
        email=payload.email,        # ใส่ได้หรือไม่ใส่ก็ได้ตาม schema
        # role ไม่ต้องส่ง → DB ใส่ default 'user' ให้เอง
    )
    db.add(user)
    try:
        # ดัน INSERT ออกไปตอนนี้เลย เพื่อให้รู้ว่าซ้ำหรือไม่
        await db.flush()
    except IntegrityError:
        # ไม่ต้อง rollback เอง ปล่อยให้ dependency จัดการเพราะมี exception เด้งออกอยู่แล้ว
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
    limit: int = 20,
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

@app.post("/channels", status_code=201)
async def create_channel(
    title: str = Form(...),
    description: str | None = Form(None),
    uploaded_files: list[UploadFile] | None = FormFile(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1) สร้าง channel
    channel = Channel(
        title=title,
        description=description,
        created_by=current_user.users_id,
        status=RoleChannel.private
    )
    db.add(channel)
    await db.flush()  # ได้ channels_id

    await db.refresh(channel, attribute_names=["channels_id", "status", "created_at"])
    is_public = (channel.status == RoleChannel.public)

    stored_files: list[dict] = []
    created_paths: list[pathlib.Path] = []

    try:
        if uploaded_files:
            if len(uploaded_files) > 10:
                raise HTTPException(status_code=400, detail="Too many files in one request")
            
            for uf in uploaded_files:
                final_path, rel_path = _build_storage_path(channel.channels_id, uf.filename)

                # 2) เขียนไฟล์ async + ตรวจขนาด
                size_bytes = await _save_upload_atomic(uf, final_path, MAX_SIZE_PER_FILE)
                created_paths.append(final_path)
                await uf.close()

                # 3) ตรวจ mime จริง (optional)
                detected = sniff_mime(final_path)
                if detected not in ALLOW_MIME:
                    final_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=400, detail=f"Unsupported file type: {detected}")

                # 4) บันทึก DB โดยเก็บ relative path
                frow = File(
                    uploaded_by=current_user.users_id,
                    channel_id=channel.channels_id,
                    original_filename=uf.filename or final_path.name,
                    storage_uri=rel_path,
                    size_bytes=size_bytes,
                )
                db.add(frow)
                await db.flush()

                # 5) เติมเอกสารเข้า RAG (Chroma) ทันที
                abs_path = UPLOAD_ROOT / rel_path
                try:
                    docs = SimpleDirectoryReader(input_files=[str(abs_path)]).load_data()
                    for d in docs:
                        d.metadata = d.metadata or {}
                        d.metadata["channel_id"] = str(channel.channels_id)
                        d.metadata["filename"] = frow.original_filename

                    # 👉 เติมเอกสารลงคอลเลกชันเดิม
                    add_documents(docs)

                except Exception as e:
                    print(f"[RAG] failed to index {abs_path}: {e}")

                resp = {
                    "files_id": frow.files_id,
                    "original_filename": frow.original_filename,
                    "size_bytes": size_bytes,
                    "mime": detected,
                }
                if is_public:
                    resp["public_url"] = f"/static/uploads/{rel_path}"
                stored_files.append(resp)

        await db.refresh(channel)

    except Exception:
        for p in created_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        raise

    return {
        "channel": {
            "channels_id": channel.channels_id,
            "title": channel.title,
            "description": channel.description,
            "status": channel.status,
        },
        "files": stored_files,
    }



@app.get("/channels/{channel_id}", response_model=ChannelOut)
async def get_channel_details(channel_id: int, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    result = await db.execute(select(Channel).where(Channel.channels_id == channel_id))
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
async def delete_channel(channel_id: int, db: AsyncSession = Depends(get_db),current_user: User = Depends(get_current_user)):
    # ดึง channel ตาม id
    result = await db.execute(
        select(Channel).where(Channel.channels_id == channel_id)
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
    
    # ลบไฟล์ในช่องทั้ง DB + ดิสก์
    file_rows = (await db.execute(select(File).where(File.channel_id == channel.channels_id))).scalars().all()
    for fr in file_rows:
        try:
            (UPLOAD_ROOT / fr.storage_uri).unlink(missing_ok=True)
        except Exception:
            pass
        await db.delete(fr)
    await db.delete(channel)
    return {"message": "Channel deleted successfully"}

@app.put("/channels/{channel_id}", response_model=ChannelListItem)
async def update_channel(
    channel_id: int,
    payload: ChannelUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Channel).where(Channel.channels_id == channel_id))
    channel = result.scalar_one_or_none()
    if channel is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    # เฉพาะ admin หรือเจ้าของเท่านั้นที่แก้ไขได้
    if current_user.role != RoleUser.admin and channel.created_by != current_user.users_id:
        raise HTTPException(status_code=403, detail="Not authorized to edit this channel")
    if payload.title is not None:
        channel.title = payload.title
    if payload.description is not None:
        channel.description = payload.description
    await db.flush()
    await db.refresh(channel)
    # นับจำนวนไฟล์ใน channel
    file_count = await db.execute(
        select(func.count(File.files_id)).where(File.channel_id == channel.channels_id)
    )
    file_count = file_count.scalar() or 0
    return ChannelListItem(
        channels_id=channel.channels_id,
        title=channel.title,
        description=channel.description,
        status=channel.status,
        created_at=channel.created_at,
        file_count=file_count,
    )

@app.post("/channels/{channel_id}/request-public", response_model=ModerationResponse, status_code=201)
async def request_make_public(
    channel_id: int,
    reason: Optional[str] = Body(None, embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1) โหลด channel
    res = await db.execute(select(Channel).where(Channel.channels_id == channel_id))
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
        channel_id=channel.channels_id,
        old_status=old_status,
        current_status=channel.status,
        event_id=event.event_id,
        message="Request submitted. Waiting for admin approval."
    )

@app.post("/channels/{channel_id}/moderate-public", response_model=AdminDecisionOut)
async def moderate_public_request(
    channel_id: int,
    payload: AdminDecisionIn,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="Admin only")
    
    res = await db.execute(select(Channel).where(Channel.channels_id == channel_id))
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

@app.put("/channels/status/{channel_id}", response_model=ChannelUpdateStatus)
async def update_channel_status(
    channel_id: int,
    new_status: RoleChannel = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Channel).where(Channel.channels_id == channel_id))
    channel = result.scalar_one_or_none()
    if channel is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    # เฉพาะ admin เท่านั้นที่แก้ไขได้
    if current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="Not authorized to edit this channel")
    
    channel.status = new_status
    await db.flush()
    await db.refresh(channel)
    
    return ChannelUpdateStatus(
        channels_id=channel.channels_id,
        status=channel.status,
    )

@app.get("/channels/public/list/", response_model=List[ChannelListPublicItem])
async def list_public_channels(
    search_by_name: str | None = Query(None, description="ค้นหาจากชื่อ"),
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Channel)
        .where(Channel.status == RoleChannel.public)
        .order_by(Channel.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    if search_by_name:
        stmt = stmt.where(Channel.title.like(f"%{search_by_name}%"))

    result = await db.execute(stmt)
    channels = result.scalars().all()

    channel_list = []
    for ch in channels:
        # ดึงรายการไฟล์ในแต่ละ channel
        file_result = await db.execute(select(File).where(File.channel_id == ch.channels_id))
        files = file_result.scalars().all()
        file_list = [
            {
                "files_id": f.files_id,
                "original_filename": f.original_filename,
                "storage_uri": f.storage_uri,
                "size_bytes": f.size_bytes,
                "created_at": f.created_at,
            }
            for f in files
        ]
        channel_list.append(ChannelListPublicItem(
            channels_id=ch.channels_id,
            title=ch.title,
            description=ch.description,
            status=ch.status,
            created_at=ch.created_at,
            files=file_list,
        ))

    return channel_list

@app.get("/channels/list/", response_model=List[ChannelListItem])
async def list_my_channels(
    search_by_name: str | None = Query(None, description="ค้นหาจากชื่อ"),
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    stmt = (
        select(Channel, func.count(File.files_id).label("file_count"))
        .outerjoin(File, File.channel_id == Channel.channels_id)
        .where(Channel.created_by == current_user.users_id)
        .group_by(Channel.channels_id)
        .order_by(Channel.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    if search_by_name:
        stmt = stmt.where(Channel.title.like(f"%{search_by_name}%"))

    result = await db.execute(stmt)
    rows = result.all()

    return [
        ChannelListItem(
            channels_id=ch.channels_id,
            title=ch.title,
            description=ch.description,
            status=ch.status,
            created_at=ch.created_at,
            file_count=file_count or 0,
        )
        for ch, file_count in rows
    ]
    
# --- แทนที่ฟังก์ชัน upload_files_only ทั้งก้อนด้วยเวอร์ชันนี้ ---
@app.post("/files/upload", status_code=201)
async def upload_files_only(
    channel_id: int = Form(...),
    files: list[UploadFile] = FormFile(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1) ตรวจสอบว่า channel มีอยู่จริง
    res = await db.execute(select(Channel).where(Channel.channels_id == channel_id))
    channel = res.scalar_one_or_none()
    if channel is None:
        raise HTTPException(status_code=404, detail="Channel not found")

    # 2) ตรวจสิทธิ์เจ้าของ channel
    if channel.created_by != current_user.users_id:
        raise HTTPException(status_code=403, detail="Not authorized to upload to this channel")

    # 3) จำกัดจำนวนไฟล์ต่อ request
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Too many files in one request")

    stored_files: list[dict] = []
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

                # เติมเอกสารลงคอลเลกชันเดิม
                add_documents(docs)

            except Exception as e:
                print(f"[RAG] failed to index {abs_path}: {e}")

            resp = {
                "files_id": frow.files_id,
                "original_filename": frow.original_filename,
                "size_bytes": size_bytes,
                "mime": detected_mime,
                "channel_id": channel_id,
            }
            if channel.status == RoleChannel.public:
                resp["public_url"] = f"/static/uploads/{rel_path}"
            stored_files.append(resp)

    except Exception:
        for p in created_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        raise

    return {"files": stored_files}


@app.delete("/files/delete/{file_id}", status_code=204)
async def delete_file(
    # channel_id: int = Path(..., gt=0),
    file_id: int = Path(..., gt=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(File).where(File.files_id == file_id))
    file = result.scalar_one_or_none()
    if file is None:
        raise HTTPException(status_code=404, detail="File not found")

    # ตรวจสอบสิทธิ์: admin หรือเจ้าของไฟล์ หรือเจ้าของ channel ที่ไฟล์อยู่
    if current_user.role != RoleUser.admin and file.uploaded_by != current_user.users_id:
        # ต้องตรวจสอบเจ้าของ channel ด้วย
        channel = await db.execute(select(Channel).where(Channel.channels_id == file.channel_id))
        channel = channel.scalar_one_or_none()
        if channel is None or channel.created_by != current_user.users_id:
            raise HTTPException(status_code=403, detail="Not authorized to delete this file")

    # ลบไฟล์จริงออกจากดิสก์
    try:
        p = UPLOAD_ROOT / file.storage_uri
        p.unlink(missing_ok=True)
    except Exception:
        pass  # ถ้าไฟล์หายไปแล้วก็ไม่เป็นไร

    # ลบเรคคอร์ดใน DB
    await db.delete(file)
    return

@app.post("/create/session", status_code=201)
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

    # 3) (ทางเลือก) ถ้าไม่อยากให้สร้างซ้ำรัว ๆ ให้เช็คก่อนว่ามีล่าสุดแล้วหรือยัง
    #    ถ้าอยากให้สร้างได้กี่รอบก็ได้ ข้ามบล็อกนี้ไปได้
    # last_sess_stmt = (
    #     select(sessions)
    #     .where(
    #         sessions.channel_id == payload.channel_id,
    #         sessions.user_id == current_user.users_id,
    #     )
    #     .order_by(sessions.created_at.desc())
    #     .limit(1)
    # )
    # last_sess_res = await db.execute(last_sess_stmt)
    # last_sess = last_sess_res.scalar_one_or_none()
    # if last_sess:
    #     # ถ้าจะ reuse session เดิม ก็ return ตัวเก่าเลย
    #     return {
    #         "session_id": last_sess.sessions_id,
    #         "channel_id": last_sess.channel_id,
    #         "user_id": last_sess.user_id,
    #         "created_at": last_sess.created_at,
    #         "reused": True,
    #     }

    # 4) สร้าง session ใหม่
    new_session = sessions(
        channel_id=payload.channel_id,
        user_id=current_user.users_id,
    )
    db.add(new_session)
    await db.flush()
    await db.refresh(new_session)

    return {
        "session_id": new_session.sessions_id,
        "channel_id": new_session.channel_id,
        "user_id": new_session.user_id,
        "created_at": new_session.created_at,
    }

@app.delete("/delete/session/{session_id}", status_code=204)
async def delete_session(session_id: int = Path(..., gt=0), db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    result = await db.execute(select(sessions).where(sessions.sessions_id == session_id))
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if current_user.role != RoleUser.admin and session.user_id != current_user.users_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this session")
    
    await db.delete(session)
    return

@app.post("/sessions/{session_id}/ollama-reply", status_code=201)
async def Talking_with_Ollama_from_document(
    session_id: int = Path(..., gt=0),
    payload: message = Body(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1) เช็คว่า session เป็นของ user นี้
    sess_stmt = (
        select(sessions)
        .where(
            sessions.sessions_id == session_id,
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

    # 3) ดึงประวัติ (จะมีหรือไม่มีก็ได้ แต่เราจะส่งให้ RAG เพื่อดึงข้อความท้าย)
    history_stmt = (
        select(chats)
        .where(chats.sessions_id == sess.sessions_id)
        .order_by(chats.created_at.asc())
        .limit(5)
    )
    history_res = await db.execute(history_stmt)
    history_rows = history_res.scalars().all()

    ai_messages = []
    for row in history_rows:
        if row.sender_type == RoleSender.user:
            ai_messages.append({"role": "user", "content": row.message})
        else:
            ai_messages.append({"role": "assistant", "content": row.message})

    # 4) เรียก RAG
    ai_text = await call_ai(ai_messages , sess.channel_id)

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
async def get_chat_history(session_id: int = Path(..., gt=0), db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    # 1) ตรวจว่า session นี้เป็นของ user นี้จริงไหม
    owned_sess = await get_owned_session(db, session_id, current_user.users_id)
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

@app.post("/chats", status_code=201)
async def create_chat(chat: message, session_id: int = Query(..., gt=0), db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    # 1) ตรวจว่า session นี้เป็นของ user นี้จริงไหม
    owned_sess = await get_owned_session(db, session_id, current_user.users_id)
    if owned_sess is None:
        # ไม่ใช่ของเขา หรือไม่มีอยู่
        raise HTTPException(status_code=403, detail="Not your session")

    # 2) ตอนนี้เรารู้แล้วว่า session นี้ของ user นี้ และรู้ด้วยว่าอยู่ channel ไหน
    #    owned_sess.channel_id คือ channel ที่ session นี้อยู่
    new_chat = chats(
        channels_id=owned_sess.channel_id,
        users_id=current_user.users_id,
        sessions_id=owned_sess.sessions_id,
        message=chat.message,
        sender_type=RoleSender.user,   # หรือจะให้ client ส่งมาก็ได้ แต่ควร validate
    )
    db.add(new_chat)
    await db.flush()
    await db.refresh(new_chat)

    return {
        "chat_id": new_chat.chat_id,
        "channel_id": new_chat.channels_id,
        "session_id": new_chat.sessions_id,
        "message": new_chat.message,
        "created_at": new_chat.created_at,
    }

