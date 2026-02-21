from fastapi import APIRouter ,Depends, HTTPException
from sqlalchemy import select
from core import logging
from db import session as db
from db.models.file_size import FileSize
from schemas.file_size import FileSizeCreate, FileSizeUpdate
from core.security import get_current_user
from core.enums import RoleUser
from db.models.user import User
router = APIRouter()

@router.get("/file-size/{file_size_id}", tags=["File Size"])
async def get_file_size(file_size_id: int, db: db.AsyncSession = Depends(db.get_db)):
    stmt = select(FileSize).where(FileSize.file_size_id == file_size_id)
    result = await db.execute(stmt)
    file_size = result.scalar_one_or_none()
    if file_size is None:
        raise HTTPException(status_code=404, detail="File size not found")
    return file_size

@router.get("/file-size/list", tags=["File Size"])
async def list_file_sizes(db: db.AsyncSession = Depends(db.get_db)):
    stmt = select(FileSize)
    result = await db.execute(stmt)
    file_sizes = result.scalars().all()
    return file_sizes

@router.post("/file-size", tags=["File Size"])
async def create_file_size(payload: FileSizeCreate, db: db.AsyncSession = Depends(db.get_db)):
   
    new_file_size = FileSize(size=payload.size)
    db.add(new_file_size)
    await db.flush()
    await db.refresh(new_file_size)
    return {
        "message": "File size created successfully",
        "file_size_id": new_file_size.file_size_id,
        "file_size": new_file_size
    }

@router.post("/file-size/update", tags=["File Size"])
async def update_file_size(payload: FileSizeUpdate, db: db.AsyncSession = Depends(db.get_db)):
    stmt = select(FileSize).where(FileSize.file_size_id == payload.id)
    result = await db.execute(stmt)
    file_size = result.scalar_one_or_none()
    
    if file_size is None:
        raise HTTPException(status_code=404, detail="File size not found")
    
    file_size.size = payload.size
    await db.flush()
    await db.refresh(file_size)
    return {
        "message": "File size updated successfully",
        "file_size_id": file_size.file_size_id,
        "file_size": file_size
    }

@router.delete("/file-size/{file_size_id}", tags=["File Size"], status_code=204)
async def delete_file_size(file_size_id: int, db: db.AsyncSession = Depends(db.get_db), current_user: User = Depends(get_current_user)):
    
    if current_user.role != RoleUser.admin:
        raise HTTPException(status_code=403, detail="ไม่มีสิทธิ์ดำเนินการ")
    
    stmt = select(FileSize).where(FileSize.file_size_id == file_size_id)
    result = await db.execute(stmt)
    file_size = result.scalar_one_or_none()
    
    if file_size is None:
        raise HTTPException(status_code=404, detail="File size not found")
    
    await db.delete(file_size)
    return