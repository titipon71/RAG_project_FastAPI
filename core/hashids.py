import logging
from typing import Optional

from hashids import Hashids

from core.config import settings

# ============================================================
#                  Hashids INITIALIZATION
# ============================================================
hasher = Hashids(salt=settings.HASH_SALT, min_length=settings.MIN_LENGTH)

def encode_id(hashed_id: int) -> str:
    logging.debug(f"Encoding ID: {hashed_id} => {hasher.encode(hashed_id)}")
    return hasher.encode(hashed_id)

def decode_id(hashed_id: str) -> Optional[int]:
    if not hashed_id:
        return None
    decoded = hasher.decode(hashed_id)
    logging.debug(f"Decoding hashed ID: {hashed_id} => {decoded}")
    if not decoded:
        return None
    return decoded[0]