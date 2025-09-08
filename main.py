# main.py
from datetime import datetime, timedelta, timezone
from typing import Optional, AsyncGenerator, List
import os, uuid, pathlib

import asyncio
from fastapi import FastAPI, APIRouter, Depends, File as FormFile, Form, UploadFile, HTTPException, status, Query
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

# ---------- Settings ----------
class Settings(BaseSettings):
    database_url: str
    secret_key: str = "dev-secret"
    access_token_expire_minutes: int = 30

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

async def get_user_by_name(db: AsyncSession, name: str) -> Optional[User]:
    stmt = select(User).where(User.name == name)
    res = await db.execute(stmt)
    return res.scalar_one_or_none()

async def get_user_by_id(db: AsyncSession, uid: int) -> Optional[User]:
    stmt = select(User).where(User.users_id == uid)
    res = await db.execute(stmt)
    return res.scalar_one_or_none()

async def authenticate_user(db: AsyncSession, name: str, password: str) -> Optional[User]:
    user = await get_user_by_name(db, name)
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

# ---------- Schemas ----------
class UserCreate(BaseModel):
    name: str
    password: str
    email: Optional[EmailStr] = None   # ตารางอนุญาตให้เป็น NULL

class UserOut(BaseModel):
    users_id: int
    name: str
    email: Optional[EmailStr] = None
    role: RoleUser
    created_at: datetime
    class Config:
        from_attributes = True

class ChannelCreate(BaseModel):
    title: str
    description: Optional[str] = None

class ChannelListItem(BaseModel):
    channels_id: int
    title: str
    description: Optional[str]
    status: RoleChannel
    created_at: datetime
    file_count: int    

# ---------- App ----------
app = FastAPI(title="FastAPI + MariaDB + JWT")

@app.on_event("startup")
async def on_startup():
    pass
#     # สร้างตารางอัตโนมัติ (เหมาะกับ dev/POC) — โปรดใช้ Alembic ในงานจริง
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)

# สมัครผู้ใช้
@app.post("/users", response_model=UserOut, status_code=201)
async def register_user(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    user = User(
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

# ออก access token ด้วย username/password จาก DB
@app.post("/auth/token")
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    user = await authenticate_user(db, form.username, form.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    # sub ต้องเป็น string ตามข้อแนะนำของ JWT
    access_token = create_access_token(data={"sub": str(user.users_id)})
    return {"access_token": access_token, "token_type": "bearer"}

# Protected endpoint
@app.get("/me")
async def read_me(current_user: User = Depends(get_current_user)):
    return {
        "users_id": current_user.users_id,
        "name": current_user.name,
        "email": current_user.email,
        "role": current_user.role,
    }

UPLOAD_DIR = pathlib.Path(r"D:\ECT\Project\FastAPITest\uploads")  # โฟลเดอร์ปลายทางในเครื่อง (ปรับตามจริง)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ตรวจไฟล์ (optional)
MAX_SIZE_PER_FILE = 20 * 1024 * 1024  # 20 MB
ALLOW_TYPES = {"application/pdf"}  # ปรับตามนโยบายคุณ
    
@app.post("/create/channel")
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

@app.delete("/delete/channels/{channel_id}")
async def delete_channel(channel_id: int, db: AsyncSession = Depends(get_db)):
    # ดึง channel ตาม id
    result = await db.execute(
        select(Channel).where(Channel.channels_id == channel_id)
    )
    channel = result.scalar_one_or_none()

    if channel is None:
        # ถ้าไม่เจอ ให้คืน 404
        raise HTTPException(status_code=404, detail="Channel not found")

    # ถ้าเจอ → ลบออก
    await db.delete(channel)


    return {"message": "Channel deleted successfully"}

@app.get("/channels/my", response_model=List[ChannelListItem])
async def list_my_channels(
    q: str | None = Query(None, description="ค้นหาจากชื่อ"),
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
    if q:
        stmt = stmt.where(Channel.title.like(f"%{q}%"))

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
