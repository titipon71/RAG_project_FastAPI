import asyncio
import logging
import pathlib
import shutil
import traceback
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Form
from fastapi import File as FastAPIFile, UploadFile
from fastapi.responses import FileResponse, Response
from llama_index.core import SimpleDirectoryReader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from core.config import settings
from core.enums import RoleChannel, RoleUser
from core.hashids import encode_id, decode_id
from core.security import get_current_user, get_optional_current_user
from db.session import get_db
from db.models.user import User
from db.models.channel import Channel
from db.models.file import File
from schemas.file import FileDetail, FileUploadResponse, FileListItem
from rag_enginex import rag_engine

# Import helpers from channels router
from routers.channels import sniff_mime, _build_storage_path, _save_upload_atomic, UPLOAD_ROOT, MAX_SIZE_PER_FILE, ALLOW_MIME

logger = logging.getLogger("uvicorn.error")

router = APIRouter()


# ============================================================
#                  FILE ROUTES
# ============================================================
@router.get("/files/list/{channel_id}", response_model=FileListItem, tags=["Files"])
async def list_files_in_channel(
    channel_id: str = Path(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    real_channel_id = decode_id(channel_id)
    if real_channel_id is None:
         raise HTTPException(status_code=404, detail="Channel ID ไม่ถูกต้อง")
     
    # 1) ตรวจสอบว่า channel มีอยู่จริง
    stmt = (
    select(Channel)
    .options(
        joinedload(Channel.creator),   # ดึง User มาพร้อมกัน
        selectinload(Channel.files)    # ดึง Files มาพร้อมกัน
    )
    .where(Channel.channels_id == real_channel_id)
    .with_for_update()
    )
    res = await db.execute(stmt)
    channel = res.scalar_one_or_none()
    if channel is None:
        raise HTTPException(status_code=404, detail="ไม่พบ Channel")
    
    is_private = channel.status in (RoleChannel.private , RoleChannel.pending)
    is_owner = channel.created_by == current_user.users_id
    is_admin = current_user.role == RoleUser.admin
        
    # 2) ตรวจสอบสิทธิ์การเข้าถึง
    if is_private and not (is_owner or is_admin):
        raise HTTPException(status_code=403, detail="ไม่มีสิทธิ์ดำเนินการเข้าถึง Channel นี้")

    # 3) ดึงรายการไฟล์
    # res = await db.execute(select(File).where(File.channel_id == real_channel_id))
    files = channel.files

    # if not files:
    #     raise HTTPException(
    #         status_code=404,
    #         detail="ไม่พบไฟล์ใน Channel นี้"
    #     )
    
    file_list: list[FileDetail] = []
    for f in files:
        hashed_file_id = encode_id(f.files_id)
        
        try:
            mime_type = sniff_mime(UPLOAD_ROOT / f.storage_uri)
        except Exception:
            mime_type = "application/octet-stream"

        file_resp = FileDetail(
            files_id=hashed_file_id,
            original_filename=f.original_filename,
            size_bytes=f.size_bytes,
            mime=mime_type,
            channel_id=channel_id,
            public_url=f"/static/uploads/{f.storage_uri}" if channel.status == RoleChannel.public else None,
            created_at=f.created_at 
        )
        file_list.append(file_resp)

    return FileListItem(files=file_list)

@router.post("/files/upload", status_code=201, response_model=FileUploadResponse, tags=["Files"])
async def upload_files_only(
    channel_id: str = Form(...),
    files: list[UploadFile] = FastAPIFile(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        real_channel_id = decode_id(channel_id)
        if real_channel_id is None:
            raise HTTPException(status_code=404, detail="Channel ID ไม่ถูกต้อง (Decode Failed)")
        
        # 1) ตรวจสอบว่า channel มีอยู่จริง
        res = await db.execute(select(Channel).where(Channel.channels_id == real_channel_id))
        channel = res.scalar_one_or_none()
        if channel is None:
            raise HTTPException(status_code=404, detail=f"ไม่พบ Channel ID: {real_channel_id}")
        
        isOwner = (channel.created_by == current_user.users_id)
        
        # 2) ตรวจสิทธิ์เจ้าของ channel
        if not isOwner:
            raise HTTPException(status_code=403, detail="ไม่มีสิทธิ์ดำเนินการอัปโหลดไปยัง Channel นี้")

        # 3) จำกัดจำนวนไฟล์ต่อ request
        if len(files) > 50:
            raise HTTPException(status_code=400, detail="จำนวนไฟล์ในหนึ่งคำขอเกินกำหนด")
        
        stored_files: list[FileDetail] = []
        created_paths: list[pathlib.Path] = []

        for uf in files:
            final_path, rel_path = _build_storage_path(real_channel_id, uf.filename)
            created_paths.append(final_path)

            # 4) เขียนไฟล์ + ตรวจขนาด
            try:
                size_bytes = await _save_upload_atomic(uf, final_path, MAX_SIZE_PER_FILE)
            finally:
                await uf.close()

            # 5) ตรวจ MIME
            detected_mime = sniff_mime(final_path)
            if detected_mime not in ALLOW_MIME:
                final_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail=f"ประเภทไฟล์ไม่รองรับ: {detected_mime}")

            # 6) บันทึก DB
            frow = File(
                uploaded_by=current_user.users_id,
                channel_id=real_channel_id,
                original_filename=uf.filename or final_path.name,
                storage_uri=rel_path,
                size_bytes=size_bytes,
            )
            db.add(frow)
            await db.flush()  # ได้ files_id
            await db.refresh(frow)

            # 7) เติมเอกสารเข้า RAG (Chroma) --- จุดเสี่ยงสูง ---
            try:
                print(f"DEBUG: กำลังส่งไฟล์ {frow.original_filename} ไปยัง RAG Engine...")
                abs_path = UPLOAD_ROOT / rel_path
                
                # เช็คว่าไฟล์มีอยู่จริงไหม
                if not abs_path.exists():
                     raise FileNotFoundError(f"หาไฟล์ไม่เจอที่: {abs_path}")

                docs = SimpleDirectoryReader(input_files=[str(abs_path)]).load_data()
                for d in docs:
                    d.metadata = d.metadata or {}
                    d.metadata["channel_id"] = str(real_channel_id)
                    d.metadata["filename"] = frow.original_filename
                    d.metadata["files_id"] = str(frow.files_id)

                # เติมเอกสารลงคอลเลกชันเดิม
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, rag_engine.add_documents, docs)
                print("DEBUG: RAG Engine ทำงานสำเร็จ")

            except Exception as rag_err:
                print(f"WARNING: RAG Error (ข้ามไปก่อน): {rag_err}")
                # ไม่ raise error เพื่อให้ upload ผ่านไปได้แม้ RAG พัง
                # แต่ print ให้เห็นใน log

            hashed_file_id = encode_id(frow.files_id)
            file_resp = FileDetail(
                files_id=hashed_file_id,
                original_filename=frow.original_filename,
                size_bytes=size_bytes,
                mime=detected_mime,
                channel_id=channel_id,
                public_url=f"/static/uploads/{rel_path}" if channel.status == RoleChannel.public else None,
                created_at=frow.created_at
            )
            stored_files.append(file_resp)

        return FileUploadResponse(files=stored_files)

    except HTTPException as he:
        # ถ้าเป็น HTTPException ให้ส่งต่อปกติ
        raise he
    except Exception as e:
        # ถ้าเป็น Error อื่นๆ ให้ปริ้นรายละเอียดออกมา
        print("="*30)
        print("!!! CRITICAL UPLOAD ERROR !!!")
        traceback.print_exc()  # ปริ้นบรรทัดที่พัง
        print(f"Error Message: {e}")
        print("="*30)
        
        # ลบไฟล์ที่ค้างอยู่
        for p in locals().get('created_paths', []):
            try:
                p.unlink(missing_ok=True)
            except:
                pass
        
        # ส่ง Error กลับไปเพื่อให้รู้ว่าเกิดอะไรขึ้น
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Upload failed. Please try again.")


@router.delete("/files/delete/{file_hash}", status_code=204, tags=["Files"])
async def delete_file(
    file_hash: str = Path(..., description="Hashed ID of the file"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        # ==========================================
        # 1. VALIDATION PHASE (ตรวจสอบความถูกต้อง)
        # ==========================================
        decoded_id = decode_id(file_hash)
        if decoded_id is None:
            raise HTTPException(status_code=404, detail="รูปแบบ ID ไฟล์ไม่ถูกต้อง")

        stmt = (
            select(File)
            .options(joinedload(File.channel))
            .where(File.files_id == decoded_id)
        )
        result = await db.execute(stmt)
        file_obj = result.scalar_one_or_none()

        if not file_obj:
            raise HTTPException(status_code=404, detail="ไม่พบไฟล์ในฐานข้อมูล")

        # ตรวจสอบสิทธิ์
        is_admin = current_user.role == RoleUser.admin
        is_owner = file_obj.uploaded_by == current_user.users_id
        # ใช้ safe navigation กัน error กรณี channel โดนลบไปก่อนแล้ว
        is_channel_owner = file_obj.channel and (file_obj.channel.created_by == current_user.users_id)

        if not (is_admin or is_owner or is_channel_owner):
            print(f"Warning: User {current_user.users_id} tried to delete file {decoded_id} without permission.")
            raise HTTPException(status_code=403, detail="ไม่มีสิทธิ์ดำเนินการลบไฟล์นี้")

        # ==========================================
        # 2. PREPARATION PHASE (เตรียม Path)
        # ==========================================
        try:
            src_path = (UPLOAD_ROOT / file_obj.storage_uri).resolve()
        except Exception:
            src_path = UPLOAD_ROOT / file_obj.storage_uri

        # Security Check
        if not str(src_path).startswith(str(UPLOAD_ROOT.resolve())):
             print(f"Critical: Path traversal attempt? {src_path}")
             raise HTTPException(status_code=400, detail="ตรวจสอบความปลอดภัยของเส้นทางไฟล์ไม่ผ่าน")

        trash_dir = settings.TRASH_DIR
        trash_filename = f"{int(datetime.now().timestamp())}_{file_obj.channel_id}_{file_obj.files_id}_{src_path.name}"
        dst_path = trash_dir / trash_filename

        # ==========================================
        # 3. EXECUTION PHASE (เริ่มลบ)
        # ==========================================

        # 3.1 ลบจาก Database
        await db.delete(file_obj)
        await db.flush() 

        # 3.2 ย้ายไฟล์จริง
        if src_path.exists():
            trash_dir.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(shutil.move, str(src_path), str(dst_path))
            print(f"DEBUG: Moved file to trash: {dst_path}")
        else:
            print(f"DEBUG: File not found on disk, skipping move: {src_path}")

        # 3.3 Clean up External Systems (RAG / AI)
        try:
            rag_engine.delete_documents_by_file_id(file_obj.files_id) 
            print(f"DEBUG: RAG docs deleted for file {file_obj.files_id}")
        except Exception as e:
            print(f"WARNING: RAG Cleanup failed: {e}")

        return Response(status_code=204)

    except HTTPException as he:
        raise he
    except Exception as e:
        # ปริ้น Error ออกมาให้เห็นชัดๆ
        print("="*30)
        print("!!! DELETE ERROR !!!")
        traceback.print_exc()
        print(f"Error Message: {e}")
        print("="*30)
        raise HTTPException(status_code=500, detail=f"Delete Failed: {str(e)}")
    
@router.get("/files/download/{file_hash}", tags=["Files"])
async def download_file(
    file_hash: str,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user), # ใช้ Optional เพื่อให้ Guest โหลดไฟล์ Public ได้
):
    # 1. Decode File ID
    file_id = decode_id(file_hash)
    if file_id is None:
        raise HTTPException(status_code=404, detail="File ID ไม่ถูกต้อง")

    # 2. Query ไฟล์พร้อม Channel เพื่อเช็คสิทธิ์
    stmt = (
        select(File)
        .options(joinedload(File.channel)) # Join เพื่อเอาสถานะ Channel มาเช็ค
        .where(File.files_id == file_id)
    )
    result = await db.execute(stmt)
    file_obj = result.scalar_one_or_none()

    if not file_obj:
        raise HTTPException(status_code=404, detail="ไม่พบไฟล์ในระบบ")

    # 3. ตรวจสอบสิทธิ์ (Security Check)
    # ---------------------------------------------------
    channel = file_obj.channel
    is_public = (channel.status == RoleChannel.public)
    
    # ถ้าไม่ใช่ Public Channel ต้อง Login และต้องเป็น (เจ้าของ หรือ Admin)
    if not is_public:
        if not current_user:
            raise HTTPException(status_code=401, detail="กรุณาเข้าสู่ระบบเพื่อดาวน์โหลดไฟล์นี้")
        
        is_owner = (file_obj.uploaded_by == current_user.users_id)
        # เช็คด้วยว่าเป็นเจ้าของ Channel หรือไม่
        is_channel_owner = (channel.created_by == current_user.users_id)
        is_admin = (current_user.role == RoleUser.admin)

        if not (is_owner or is_channel_owner or is_admin):
            raise HTTPException(status_code=403, detail="ไม่มีสิทธิ์ดาวน์โหลดไฟล์นี้")

    # 4. เตรียม Path ไฟล์
    # ---------------------------------------------------
    file_path = UPLOAD_ROOT / file_obj.storage_uri
    
    # ตรวจสอบว่าไฟล์มีอยู่จริงใน Disk ไหม
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="ไฟล์สูญหายจาก Server (File Not Found on Disk)")

    # 5. ส่งไฟล์กลับ (FileResponse จะจัดการเรื่อง Header ให้เอง)
    # media_type ถ้าไม่ระบุ Browser จะพยายามเดาเอง
    # filename=... จะทำให้ Browser เด้งหน้าต่าง Save As เป็นชื่อไฟล์เดิม
    return FileResponse(
        path=file_path, 
        filename=file_obj.original_filename,
        media_type="application/octet-stream" # บังคับให้ Download (ถ้าอยากให้เปิดใน Browser ให้แก้ตาม MIME type)
    )
