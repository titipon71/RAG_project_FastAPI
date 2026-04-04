from schemas.base import ORMBase


# --- Events schemas ---
class EventsAsReadRequest(ORMBase):
    event_id: int
    type: str = "user"  # "user" หรือ "admin" เพื่อระบุว่าใครเป็นฝ่ายอ่าน