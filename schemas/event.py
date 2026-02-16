from schemas.base import ORMBase


# --- Events schemas ---
class EventsAsReadRequest(ORMBase):
    event_id: int