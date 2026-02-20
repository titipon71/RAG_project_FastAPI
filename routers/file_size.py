from fastapi import APIRouter ,Depends, HTTPException
from sqlalchemy import select
from core import logging
from db import session as db
from db.models.file_size import FileSize
from schemas.file_size import FileSizeCreate

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
async def create_file_size(playload: FileSizeCreate, db: db.AsyncSession = Depends(db.get_db)):
    stmt = (select(FileSize).where(FileSize.file_size_id == playload.id))
    result = await db.execute(stmt)
    existing_file_size = result.scalar_one_or_none()
    if existing_file_size:
        raise HTTPException(status_code=400, detail="File size with this ID already exists")
    
    new_file_size = FileSize(size=playload.size)
    db.add(new_file_size)
    await db.flush()
    await db.refresh(new_file_size)
    return new_file_size
