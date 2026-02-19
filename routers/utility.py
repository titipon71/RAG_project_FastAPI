import base64
from typing import Optional

from fastapi import APIRouter, Body, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.templating import Jinja2Templates
from scalar_fastapi import get_scalar_api_reference

from core.config import settings
from core.hashids import hasher
from rag_enginex import rag_engine

router = APIRouter()

templates = Jinja2Templates(directory="templates")


# ============================================================
#                  BASIC / HEALTH ROUTES
# ============================================================
@router.get("/healthz", response_class=PlainTextResponse, tags=["System & Utility"])
def healthz_get():
    return "ok"

@router.head("/healthz", tags=["System & Utility"])
def healthz_head():
    return Response(status_code=200)

base64_icon = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    image_bytes = base64.b64decode(base64_icon)
    
    return Response(content=image_bytes, media_type="image/png")

@router.head("/", tags=["System & Utility"])
def root_head():
    return Response(status_code=200)

@router.get("/hashids-demo", response_class=HTMLResponse, tags=["System & Utility"])
async def read_root(request: Request):
    return templates.TemplateResponse("hashids_demo.html", {
        "request": request,
        "salt_preview": settings.HASH_SALT[:3] + "***"
    })

# 3. POST Route: รวม Logic ทั้ง Encode และ Decode
@router.post("/hashids-demo", response_class=HTMLResponse, tags=["System & Utility"])
async def process_hashids(
    request: Request,
    action: str = Form(...),              # ตัวบอกว่ากดปุ่มไหน (encode หรือ decode)
    number_input: Optional[int] = Form(None), # รับค่าตัวเลข (ถ้ามี)
    hash_input: Optional[str] = Form(None)    # รับค่า Hash (ถ้ามี)
):
    result_encode = None
    result_decode = None
    error_msg = None

    # Logic: ตรวจสอบว่า action คืออะไร
    if action == "encode" and number_input is not None:
        result_encode = hasher.encode(number_input)
        
    elif action == "decode" and hash_input:
        decoded = hasher.decode(hash_input)
        if decoded:
            result_decode = decoded[0]
        else:
            error_msg = "ไม่สามารถถอดรหัสได้ (Invalid Hash)"

    # ส่งค่ากลับไปที่ template เดิม
    return templates.TemplateResponse("hashids_demo.html", {
        "request": request,
        "salt_preview": settings.HASH_SALT[:3] + "***",
        "action": action,             # ส่งกลับไปเพื่อเช็คว่าเพิ่งทำอะไรเสร็จ
        "number_input": number_input, # ส่งค่าเดิมกลับไปแสดงในช่อง
        "hash_input": hash_input,     # ส่งค่าเดิมกลับไปแสดงในช่อง
        "result_encode": result_encode,
        "result_decode": result_decode,
        "error_msg": error_msg
    })

@router.get("/rapidoc", include_in_schema=False)
async def rapidoc():
    html_content = """
    <!doctype html>
    <html>
    <head>
        <title>RapiDoc</title>
        <meta charset="utf-8">
        <script type="module" src="https://unpkg.com/rapidoc/dist/rapidoc-min.js"></script>
    </head>
    <body>
        <rapi-doc 
            spec-url="/openapi.json"
            theme="dark" 
            allow-try="true"
            allow-server-selection="false"
            show-header="false"
            render-style="focused"
            primary-color="#34A853"
            nav-spacing="relaxed"
            show-method-in-nav-bar="as-colored-block"
            font-size="largest"
            show-info="false"
            allow-spec-url-load="false"
        > 
        </rapi-doc>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@router.get("/robots.txt", response_class=PlainTextResponse, include_in_schema=False)
def robots_txt():
    date = """User-agent: *
Disallow: /"""
    return date


# ============================================================
#                  DEBUG / UTIL ROUTES
# ============================================================
@router.post("/debug", tags=["System & Utility"])
def debug_endpoint(
    channel_id: int = Body(..., embed=True)
):
    try:
        rag_engine.debug_list_docs_by_channel(channel_id=channel_id)
        return {"message": "Debug payload received"}
    except Exception as e:
        return {"error": str(e)}

# @router.get("/debug-user-info")
# async def debug_user_info(db: AsyncSession = Depends(get_db)):
#     result = await db.execute(select(User))
#     users = result.scalars().all()
#     user_list = []
#     for u in users:
#         user_list.append({
#             "users_id": u.users_id,
#             "username": u.username,
#             "role": u.role,
#             "created_at": u.created_at,
#         })
#     return user_list

@router.get("/callback", tags=["System & Utility"], response_class=HTMLResponse)
async def get_callback(code: str, state: str):
    return templates.TemplateResponse("callback.html", {
        "request": Request(scope={"type": "http", "method": "GET", "path": "/callback"}),
        "code": code,
        "state": state
    })