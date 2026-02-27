
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from db.models.file_size import FileSize


async def get_or_create_file_size(
    db: AsyncSession,
    size: int,
) -> FileSize:

    stmt = select(FileSize).where(FileSize.size == size)
    result = await db.execute(stmt)
    file_size = result.scalar_one_or_none()

    if file_size:
        return file_size

    file_size = FileSize(size=size)
    db.add(file_size)

    try:
        await db.flush()
        return file_size

    except IntegrityError:
        await db.rollback()

        result = await db.execute(
            select(FileSize).where(FileSize.size == size)
        )
        return result.scalar_one()