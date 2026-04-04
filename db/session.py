from typing import AsyncGenerator
from sqlalchemy import event
import traceback
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from core.config import settings

# ============================================================
#                      DB ENGINE & SESSION
# ============================================================
engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
)

# --- Debug: ติดตาม connection ที่ยังค้างอยู่ ---
        
if False:
    _checked_out = {}  # connection id → stack trace

    @event.listens_for(engine.sync_engine, "checkout")
    def on_checkout(dbapi_conn, conn_record, conn_proxy):
        print("🔵 checkout", id(dbapi_conn))
        _checked_out[id(dbapi_conn)] = traceback.format_stack()

    @event.listens_for(engine.sync_engine, "checkin")
    def on_checkin(dbapi_conn, conn_record):
        print("🟢 checkin", id(dbapi_conn))
        _checked_out.pop(id(dbapi_conn), None)

    @event.listens_for(engine.sync_engine, "close")
    def on_close(dbapi_conn, conn_record):
        stack = _checked_out.pop(id(dbapi_conn), None)
        if stack:
            print("⚠️  Connection closed WITHOUT checkin — likely leaked here:")
            print("".join(stack))

SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
