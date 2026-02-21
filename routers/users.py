import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Path, Body
import httpx
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from core.enums import RoleUser
from core.security import get_current_user, verify_password
from db.session import get_db
from db.models.user import User
from schemas.user import UserCreate, UserOut, UserUpdate, UserPasswordUpdate, SSOUserInfo
from services.user_service import get_user_by_id

logger = logging.getLogger("uvicorn.error")

router = APIRouter()

# ============================================================
#                 SSO USER ROUTES 
# ============================================================
@router.post("/user/kmutnb-sso/info", tags=["Users"])
async def get_sso_user_info(payload: SSOUserInfo):
    async with httpx.AsyncClient() as client:
        try:
            headers = {"Authorization": f"Bearer {payload.sso_access_token}"}
            response = await client.get("https://sso.kmutnb.ac.th/resources/userinfo", headers=headers)
            response.raise_for_status()
            data = response.json()
            return data
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch SSO user info: {e}")
            raise HTTPException(status_code=502, detail="ไม่สามารถเชื่อมต่อกับ SSO ได้")


# ============================================================
#                  USER ROUTES (CRUD + ROLE)
# ============================================================

@router.post("/users", response_model=UserOut, status_code=201, tags=["Users"])
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
        raise HTTPException(status_code=409, detail="ชื่อหรืออีเมลนี้มีอยู่ในระบบแล้ว")

    # โหลดค่าที่ DB เติมให้ (เช่น id/created_at/role default)
    await db.refresh(user)
    return user

# Read: Get user by id
@router.get("/users/{user_id}", response_model=UserOut, tags=["Users"])
async def get_user_by_id_api(
    user_id: int = Path(..., gt=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้งาน")
    # เฉพาะ admin หรือเจ้าของเท่านั้นที่ดูได้
    if current_user.role != RoleUser.admin and current_user.users_id != user_id:
        raise HTTPException(status_code=403, detail="ไม่มีสิทธิ์ดำเนินการ")
    return user

# Update: Update user info (username, name, email)
@router.put("/users/{user_id}", response_model=UserOut, tags=["Users"])
async def update_user(
    user_id: int,
    payload: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้งาน")
    if current_user.role != RoleUser.admin and current_user.users_id != user_id:
        raise HTTPException(status_code=403, detail="ไม่มีสิทธิ์ดำเนินการ")
    if payload.username is not None:
        user.username = payload.username
    if payload.name is not None:
        user.name = payload.name
    if payload.email is not None:
        user.email = payload.email
    try:
        await db.flush()
    except IntegrityError:
        raise HTTPException(status_code=409, detail="ชื่อผู้ใช้, ชื่อ หรืออีเมลนี้มีอยู่ในระบบแล้ว")
    await db.refresh(user)
    return user

@router.put("/users/password/{user_id}", status_code=204, tags=["Users"])
async def update_user_password(
    user_id: int = Path(..., gt=0),
    payload: UserPasswordUpdate = Body(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้งาน")
    if payload.new_password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="รหัสผ่านใหม่และการยืนยันรหัสผ่านไม่ตรงกัน")
    if current_user.role != RoleUser.admin and current_user.users_id != user_id:
        raise HTTPException(status_code=403, detail="ไม่มีสิทธิ์ดำเนินการ")
    if not verify_password(payload.old_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="รหัสผ่านเดิมไม่ถูกต้อง")
    user.hashed_password = payload.new_password
    await db.flush()
    return

# Delete: Delete user
@router.delete("/users/{user_id}", status_code=204, tags=["Users"])
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้งาน")
    if current_user.role != RoleUser.admin and current_user.users_id != user_id:
        raise HTTPException(status_code=403, detail="ไม่มีสิทธิ์ดำเนินการ")
    await db.delete(user)
    return

@router.put("/user/role/{user_id}/{new_role}", response_model=UserOut, tags=["Users"])
async def update_user_role(
    user_id: int = Path(..., gt=0),
    new_role: RoleUser = Path(...),
    secret_key: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="ไม่มีสิทธิ์ดำเนินการ")

    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้งาน")

    if secret_key != "4+2*3=24":
        raise HTTPException(status_code=403, detail="secret_key ไม่ถูกต้อง")
    logger.info(f"Admin {current_user.users_id} changed user {user_id} role to {new_role}")
    user.role = new_role
    await db.flush()
    await db.refresh(user)

    return user

# Read: List all users (admin only)
@router.get("/users/list/", response_model=List[UserOut], tags=["Users"])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="ไม่มีสิทธิ์ดำเนินการ")
    stmt = select(User).options(joinedload(User.file_size)).offset(skip).limit(limit)
    result = await db.execute(stmt)
    users = result.scalars().all()
    return users

# Protected endpoint
@router.get("/get/userinfo/bytoken", tags=["Users"])
async def get_user_by_token(current_user: User = Depends(get_current_user)):
    return {
        "users_id": current_user.users_id,
        "username": current_user.username,
        "name": current_user.name,
        "account_type": current_user.account_type,
        "file_size_id": current_user.file_size_id,
        "file_size": current_user.file_size.size if current_user.file_size else None,
        "email": current_user.email,
        "role": current_user.role,
    }


@router.put('/role/update/{user_id}/{new_role}', response_model=UserOut, tags=["Users"])
async def update_role(
    user_id: int = Path(..., gt=0),
    new_role: RoleUser = Path(...),
    db: AsyncSession = Depends(get_db),
):
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้งาน")
    user.role = new_role
    await db.flush()
    await db.refresh(user)

    return user

@router.get('/role/list', tags=["Users"])
async def get_all_roles():
    return list(RoleUser)