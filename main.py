# main.py
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from scalar_fastapi import get_scalar_api_reference

from core.config import settings
from core.tag import tags_metadata
from core.cors import ALLOWED_ORIGINS, ALLOW_ORIGIN_REGEX
from core.logging import apply_custom_logging
import db.models
# Routers
from routers import auth, users, file_size, channels, files, session, events, statistics, api_key, utility

# ============================================================
#                  APP INITIALIZATION
# ============================================================
app = FastAPI(title="KMUTNBLM (FastAPI + MariaDB + JWT)", openapi_tags=tags_metadata)
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
app.include_router(file_size.router)
app.include_router(channels.router)
app.include_router(files.router)
app.include_router(session.router)
app.include_router(events.router)
app.include_router(statistics.router)
app.include_router(api_key.router)
app.include_router(utility.router)

# ---------- Startup Event ----------
@app.on_event("startup")
async def startup_event():
    apply_custom_logging()

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