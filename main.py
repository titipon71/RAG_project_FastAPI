# main.py
from datetime import datetime, timedelta, timezone
from typing import Optional, AsyncGenerator, List
from urllib import response
import uuid, pathlib
import aiofiles
from fastapi import Body, FastAPI, APIRouter, Depends, File as FormFile, Form, UploadFile, HTTPException, status, Query, Path , Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel , EmailStr
from pydantic_settings import BaseSettings
from sqlalchemy import String, func, select ,Enum as SAEnum, ForeignKey, text
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.exc import IntegrityError
import enum
from sqlalchemy.dialects.mysql import INTEGER as MyInt, ENUM as MyEnum
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# import fastapi
# import asyncio
# import fastapi_swagger_dark as fsd
# from fastapi_limiter import FastAPILimiter
# from fastapi_limiter.depends import RateLimiter
# from fastapi.responses import JSONResponse

# ---------- Settings ----------
class Settings(BaseSettings):
    database_url: str
    secret_key: str = "dev-secret"
    access_token_expire_minutes: int = 720

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

class RoleSender(str, enum.Enum):
    user = "user"
    AI = "AI"
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
    create_at: Mapped[datetime] = mapped_column("created_at", server_default=func.current_timestamp(), nullable=False)

class chats(Base):
    __tablename__ = "chats"
    chats_id: Mapped[int] = mapped_column("chats_id", MyInt(unsigned=True), primary_key=True, autoincrement=True)
    channels_id: Mapped[int] = mapped_column("channels_id", MyInt(unsigned=True), nullable=False)
    users_id: Mapped[int] = mapped_column("users_id", MyInt(unsigned=True), nullable=False)
    sessions_id: Mapped[int] = mapped_column("sessions_id", MyInt(unsigned=True), nullable=False)
    message: Mapped[str] = mapped_column("message", String(2000), nullable=False)
    sender_type: Mapped[RoleSender] = mapped_column("sender_type", MyEnum(RoleSender), nullable=False ,)
    created_at: Mapped[datetime] = mapped_column("created_at", server_default=func.current_timestamp(), nullable=False)

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

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def hash_password(plain: str) -> str:
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
    
class message(BaseModel):
    message: str
# ---------- App ----------

app = FastAPI(title="FastAPI + MariaDB + JWT")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # ปรับเป็นโดเมนที่ต้องการอนุญาต
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

@app.get("/",status_code=200)
async def root():
    return {"message": "Welcome to FastAPI + MariaDB + JWT"}

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
        hashed_password=hash_password(payload.password),
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
    user.hashed_password = hash_password(payload.new_password)
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

@app.put("/user/role/{user_id}/{new_role}", response_model=UserRoleUpdate)
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
UPLOAD_DIR = pathlib.Path(r"D:\ECT\Project\FastAPITest\uploads")  # โฟลเดอร์ปลายทางในเครื่อง (ปรับตามจริง)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ตรวจไฟล์ (optional)
MAX_SIZE_PER_FILE = 20 * 1024 * 1024  # 20 MB
ALLOW_TYPES = {"application/pdf"}  # ปรับตามนโยบายคุณ
    
@app.post("/channels", status_code=201)
async def create_channel(
    title: str = Form(...),
    description: str | None = Form(None),
    uploaded_files: list[UploadFile] | None = FormFile(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1) สร้าง channel
    channel = Channel(title=title, description=description, created_by=current_user.users_id)
    db.add(channel)
    await db.flush()  # ให้ได้ channels_id

    stored_files: list[dict] = []
    created_paths: list[pathlib.Path] = []

    try:
        if uploaded_files:
            for uf in uploaded_files:
                # ตรวจชนิดไฟล์ (header)
                if uf.content_type not in ALLOW_TYPES:
                    raise HTTPException(status_code=400, detail=f"Unsupported file type: {uf.content_type}")

                # สร้างชื่อไฟล์ปลอดภัย
                suffix = pathlib.Path(uf.filename or "").suffix
                safe_name = f"{uuid.uuid4().hex}{suffix}"
                disk_path = UPLOAD_DIR / safe_name

                # เขียนไฟล์ลงดิสก์ (จะให้ดีใช้ aiofiles)
                size_counter = 0
                try:
                    with open(disk_path, "wb") as f:
                        while True:
                            chunk = await uf.read(1024 * 1024)
                            if not chunk:
                                break
                            size_counter += len(chunk)
                            if size_counter > MAX_SIZE_PER_FILE:
                                raise HTTPException(status_code=413, detail=f"File too large: {uf.filename}")
                            f.write(chunk)
                    created_paths.append(disk_path)
                finally:
                    await uf.close()

                # (ถ้ามี StaticFiles ให้ map เป็น URL จริง เช่น /static/uploads/<safe_name>)
                storage_uri = str(disk_path)  # เก็บเป็น path ภายในให้ชัดเจนก่อน

                file_row = File(
                    uploaded_by=current_user.users_id,
                    channel_id=channel.channels_id,
                    original_filename=uf.filename or safe_name,
                    storage_uri=storage_uri,
                    size_bytes=size_counter,
                )
                db.add(file_row)
                await db.flush()  # เอา files_id

                stored_files.append({
                    "files_id": file_row.files_id,
                    "original_filename": file_row.original_filename,
                    "storage_uri": file_row.storage_uri,
                    "size_bytes": file_row.size_bytes,
                })

        # โหลดค่า default (เช่น status) ให้ชัวร์ก่อนส่งคืน
        await db.refresh(channel)

    except Exception:
        # ถ้า error ใดๆ ให้ลบไฟล์ที่เขียนไปแล้วออก เพื่อไม่ทิ้ง orphan
        for p in created_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        raise  # ส่ง error ต่อให้ FastAPI จัดการ

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
    
    # ตรวจสอบสิทธิ์การเข้าถึง
    if channel.status == RoleChannel.private and channel.created_by != current_user.users_id and current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="Not authorized to access this channel")
    
    # ดึงรายการไฟล์ที่อยู่ใน channel นี้
    result = await db.execute(select(File).where(File.channel_id == channel_id))
    files = result.scalars().all()
    
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
    
    # ถ้าเจอ → ลบออก
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

@app.put("/channels/status/{channel_id}", response_model=ChannelUpdateStatus)
async def update_channel_status(
    channel_id: int,
    new_status: RoleChannel = Path(...),
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
    
@app.post("/files/upload", status_code=201)
async def upload_files_only(
    channel_id: int = Form(None),          
    files: list[UploadFile] = FormFile(...),      # รับหลายไฟล์
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1) ตรวจสอบว่ามี channel จริงไหม
    res = await db.execute(select(Channel).where(Channel.channels_id == channel_id))
    channel = res.scalar_one_or_none()
    if channel is None:
        raise HTTPException(status_code=404, detail="Channel not found")

    # 2) ตรวจสอบสิทธิ์เจ้าของ channel
    if channel.created_by != current_user.users_id:
        raise HTTPException(status_code=403, detail="Not authorized to upload to this channel")

    stored_files: list[dict] = []
    created_paths: list[pathlib.Path] = []

    try:
        for uf in files:
            # ตรวจชนิดไฟล์ตามนโยบาย
            if uf.content_type not in ALLOW_TYPES:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {uf.content_type}")

            # ตั้งชื่อไฟล์ให้ปลอดภัย
            suffix = pathlib.Path(uf.filename or "").suffix
            safe_name = f"{uuid.uuid4().hex}{suffix}"
            disk_path = UPLOAD_DIR / safe_name

            try:
                size_bytes = await _save_upload_to_disk(uf, disk_path, MAX_SIZE_PER_FILE)
                created_paths.append(disk_path)
            finally:
                await uf.close()

            # เก็บเป็น path ภายใน (หรือจะแมปเป็น URL เสิร์ฟจริงก็ได้)
            storage_uri = str(disk_path)

            # บันทึก DB
            frow = File(
                uploaded_by=current_user.users_id,
                channel_id=channel_id,
                original_filename=uf.filename or safe_name,
                storage_uri=storage_uri,
                size_bytes=size_bytes,
            )
            db.add(frow)
            await db.flush()  # ได้ files_id

            # ถ้าคุณ mount /static ตามด้านบน จะทำ public URL ง่ายๆ ได้แบบนี้:
            public_url = f"/static/uploads/{safe_name}"

            stored_files.append({
                "files_id": frow.files_id,
                "original_filename": frow.original_filename,
                "storage_uri": storage_uri,   # path ภายใน
                # "public_url": public_url,     # URL สำหรับเข้าถึง (ถ้าต้องการ)
                "size_bytes": size_bytes,
                "content_type": uf.content_type,
                "channel_id": channel_id,
            })

    except Exception:
        # ลบไฟล์ที่เขียนไปแล้วถ้ามี error
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
        p = pathlib.Path(file.storage_uri)
        p.unlink(missing_ok=True)
    except Exception:
        pass  # ถ้าไฟล์หายไปแล้วก็ไม่เป็นไร

    # ลบเรคคอร์ดใน DB
    await db.delete(file)
    return

@app.post("/create/session", status_code=201)
async def create_session(session: sessionCreate, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    chackcidinbd = await db.execute(select(Channel).where(Channel.channels_id == session.channel_id))
    chackcidinbd = chackcidinbd.scalar_one_or_none()
    if chackcidinbd is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    new_session = sessions(
        channel_id=session.channel_id,
        user_id=current_user.users_id,
    )
    db.add(new_session)
    await db.flush()
    await db.refresh(new_session)
    return {"session_id": new_session.sessions_id}

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

@app.post("/chats", status_code=201)
async def create_chat(chat: chatCreate, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    new_chat = chats(
        channels_id=chat.channels_id,
        users_id=current_user.users_id,
        sessions_id=chat.sessions_id,
        message=chat.message,
        sender_type=chat.sender_type,
    )
    db.add(new_chat)
    await db.flush()
    await db.refresh(new_chat)
    return {"chat_id": new_chat.chats_id}

