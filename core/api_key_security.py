import hashlib
import logging
from typing import Tuple

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db
from db.models.user import User
from db.models.api_key import ApiKey

logger = logging.getLogger("uvicorn.error")

# ============================================================
#               Security Dependency public API Key
# ============================================================
bearer_scheme = HTTPBearer(auto_error=False)

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

async def get_api_key_context(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db)
) -> Tuple[User, ApiKey]:
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=403, detail="API Key is missing")

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=403, detail="Invalid authorization scheme")

    key = credentials.credentials

    hashed = hash_key(key)

    stmt = select(ApiKey).where(ApiKey.key_hash == hashed, ApiKey.is_active == True)
    result = await db.execute(stmt)
    api_key_obj = result.scalar_one_or_none()

    # Backward compatibility for old records that stored raw key in key_hash.
    # If found, migrate in-place to hashed format.
    if not api_key_obj:
        legacy_stmt = select(ApiKey).where(ApiKey.key_hash == key, ApiKey.is_active == True)
        legacy_result = await db.execute(legacy_stmt)
        api_key_obj = legacy_result.scalar_one_or_none()
        if api_key_obj:
            api_key_obj.key_hash = hashed
            await db.flush()
            logger.warning("Migrated legacy API key storage format to hashed key_id=%s", api_key_obj.key_id)

    if not api_key_obj:
        raise HTTPException(status_code=401, detail="Invalid or Inactive API Key")

    user_stmt = select(User).where(User.users_id == api_key_obj.user_id)
    user_res = await db.execute(user_stmt)
    user = user_res.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user, api_key_obj