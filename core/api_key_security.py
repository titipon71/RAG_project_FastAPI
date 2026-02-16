import hashlib
from typing import Tuple

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db
from db.models.user import User
from db.models.api_key import ApiKey

# ============================================================
#               Security Dependency public API Key
# ============================================================
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

async def get_api_key_context(
    key: str = Depends(api_key_header),
    db: AsyncSession = Depends(get_db)
) -> Tuple[User, ApiKey]:
    if not key:
        raise HTTPException(status_code=403, detail="API Key is missing")

    hashed = hash_key(key)

    stmt = select(ApiKey).where(ApiKey.key_hash == hashed, ApiKey.is_active == True)
    result = await db.execute(stmt)
    api_key_obj = result.scalar_one_or_none()

    if not api_key_obj:
        raise HTTPException(status_code=401, detail="Invalid or Inactive API Key")

    user_stmt = select(User).where(User.users_id == api_key_obj.user_id)
    user_res = await db.execute(user_stmt)
    user = user_res.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user, api_key_obj