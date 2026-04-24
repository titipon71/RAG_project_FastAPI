# main.py
import asyncio
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from scalar_fastapi import get_scalar_api_reference

from core.sse_manager import sse_manager
from core.config import settings
from core.tag import tags_metadata
from core.cors import ALLOWED_ORIGINS, ALLOW_ORIGIN_REGEX
from core.logging import apply_custom_logging
import db.models
# Routers
from routers import account_type, auth, users, file_size, channels, files, session, events, statistics, api_key, utility
from db.session import engine
from contextlib import asynccontextmanager
from rag_enginex import rag_engine

# ============================================================
#                  APP INITIALIZATION
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    apply_custom_logging()
    logger = logging.getLogger("uvicorn.error")
    
    # ── Startup ──
    logger.info("Starting up...")
    if rag_engine:
        rag_engine.summary_worker.start()
        logger.info("SummaryWorker started")
    else:
        logger.warning("RAG Engine is not available, skipping SummaryWorker start")
    logger.info("Startup complete")
    
    yield
    
    # ── Shutdown ──
    logger.info("Shutting down...")
    
    await sse_manager.shutdown()
    await asyncio.sleep(0.5)
    
    if rag_engine:
        await rag_engine.summary_worker.shutdown()
        logger.info("SummaryWorker shut down")
    await engine.dispose()
    logger.info("Database engine disposed")
    logger.info("Shutdown complete")
    
app = FastAPI(
    title="KMUTNBLM (FastAPI + MariaDB + JWT)",
    root_path="/fastapi",
    openapi_url="/openapi.json",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)
templates = Jinja2Templates(directory="templates")

# ---------- Static Files ----------
UPLOAD_ROOT = settings.upload_root
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/static/uploads", StaticFiles(directory=UPLOAD_ROOT), name="uploads")
app.mount("/static/templates", StaticFiles(directory="templates"), name="templates_static")

# ---------- Middleware ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Register Routers ----------
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(account_type.router)
app.include_router(file_size.router)
app.include_router(channels.router)
app.include_router(files.router)
app.include_router(session.router)
app.include_router(events.router)
app.include_router(statistics.router)
app.include_router(api_key.router)
app.include_router(utility.router)


    
# # ---------- Startup Event ----------
# @app.on_event("startup")
# async def startup_event():
#     apply_custom_logging()

# ---------- Root / Docs ----------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/scalar", include_in_schema=False)
async def scalar_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title="API Docs",
    )
