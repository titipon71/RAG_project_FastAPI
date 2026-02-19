import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.enums import RoleUser
from core.security import create_access_token
from db.session import get_db
from db.models.user import User
from schemas.auth import SSOCodeRequest
from services.user_service import authenticate_user

logger = logging.getLogger("uvicorn.error")

router = APIRouter()


# ============================================================
#                  AUTH ROUTES
# ============================================================
@router.post("/auth/token", tags=["Authentication"])
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    user = await authenticate_user(db, form.username, form.password)
    if not user:
        raise HTTPException(status_code=400, detail="ชื่อผู้ใช้งานหรือรหัสผ่านไม่ถูกต้อง")
    # sub ต้องเป็น string ตามข้อแนะนำของ JWT
    access_token = create_access_token(data={"sub": str(user.users_id)})
    return {"access_token": access_token, "token_type": "bearer"}

# --- KMUTNB SSO ---
# Token Request
@router.post("/auth/sso/kmutnb", tags=["Authentication"])
async def sso_kmutnb(payload: SSOCodeRequest, db: AsyncSession = Depends(get_db)):
    try:
        # =========================
        # 1) ขอ Token จาก SSO
        # =========================
        try:
            async with httpx.AsyncClient() as client:
                token_response = await client.post(
                    settings.SSO_TOKEN_URL,
                    auth=(settings.SSO_CLIENT_ID, settings.SSO_CLIENT_SECRET),
                    data={
                        "grant_type": "authorization_code",
                        "code": payload.code,
                        "redirect_uri": settings.SSO_REDIRECT_URI,
                    }
                )
        except httpx.RequestError as e:
            logger.error(f"SSO Token Request Error: {e}", exc_info=True)
            raise HTTPException(status_code=502, detail="ไม่สามารถเชื่อมต่อ SSO Server ได้")

        if token_response.status_code != 200:
            logger.error(f"SSO Token Failed: {token_response.text}")
            raise HTTPException(status_code=400, detail="SSO token request failed")

        try:
            token_json = token_response.json()
        except Exception as e:
            logger.error(f"Invalid Token JSON: {e}", exc_info=True)
            raise HTTPException(status_code=502, detail="SSO ตอบกลับข้อมูล token ผิดรูปแบบ")

        access_token = token_json.get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail=f"ไม่ได้รับ access_token จาก SSO: {token_json}")

        # =========================
        # 2) ดึงข้อมูลผู้ใช้จาก SSO
        # =========================
        try:
            async with httpx.AsyncClient() as client:
                user_info_res = await client.get(
                    settings.SSO_USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"}
                )
        except httpx.RequestError as e:
            logger.error(f"SSO UserInfo Request Error: {e}", exc_info=True)
            raise HTTPException(status_code=502, detail="ไม่สามารถดึงข้อมูลผู้ใช้จาก SSO ได้")

        if user_info_res.status_code != 200:
            logger.error(f"SSO UserInfo Failed: {user_info_res.text}")
            raise HTTPException(status_code=400, detail=f"SSO userinfo request failed: {user_info_res.text}")

        try:
            sso_data = user_info_res.json()
        except Exception as e:
            logger.error(f"Invalid UserInfo JSON: {e}", exc_info=True)
            raise HTTPException(status_code=502, detail="SSO ตอบกลับข้อมูลผิดรูปแบบ")

        username = sso_data.get("profile", {}).get("username")
        if not username:
            raise HTTPException(status_code=400, detail=f"ไม่พบ username ในข้อมูล SSO: {sso_data}")

        # =========================
        # 3) ค้นหาหรือสร้าง User
        # =========================
        try:
            stmt = select(User).where(User.username == username)
            result = await db.execute(stmt)
            user = result.scalar_one_or_none()

            if not user:
                new_user = User(
                    username=username,
                    name=sso_data.get("profile", {}).get("name_en"),
                    email=sso_data.get("profile", {}).get("email"),
                    account_type=sso_data.get("profile", {}).get("account_type"),
                    hashed_password=username,  # แนะนำให้เปลี่ยนใน production
                    role=RoleUser.user
                )
                db.add(new_user)
                await db.flush()
                await db.refresh(new_user)
                user = new_user
            else:
                user.name = sso_data.get("profile", {}).get("name_en")
                user.email = sso_data.get("profile", {}).get("email")
                user.account_type = sso_data.get("profile", {}).get("account_type")
                await db.flush()

        except IntegrityError:
            logger.warning(f"Duplicate user from SSO | username={username}")
            raise HTTPException(status_code=409, detail="ข้อมูลผู้ใช้ซ้ำกับที่มีในระบบ")

        except SQLAlchemyError as e:
            logger.error(f"SSO DB Error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="เกิดข้อผิดพลาดในฐานข้อมูล")

        # =========================
        # 4) Return Response
        # =========================
        return {
            "message": "SSO login successful",
            "user_id": user.users_id,
            "username": user.username,
            "account_type": user.account_type,
            "local_access_token": create_access_token(
                data={"sub": str(user.users_id)}
            ),
            "sso_access_token": access_token
        }

    except HTTPException:
        logger.warning("Caught HTTPException in SSO login")
        raise

    except Exception as e:
        logger.error(f"Unexpected SSO Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )