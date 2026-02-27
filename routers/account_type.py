from fastapi import APIRouter ,Depends, HTTPException
from sqlalchemy import select
from core import logging
from db.models import account_type
from db.session import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from db.models.account_type import AccountType
import traceback
from schemas.account_types import AccountTypeResponse, AccountTypeUpdateSizeRequest
from sqlalchemy.orm import joinedload

router = APIRouter()

@router.get("/account-types", tags=["Account Types"] , response_model=list[AccountTypeResponse])
async def list_account_types(db: AsyncSession = Depends(get_db)):
    try:
         stmt = select(AccountType).options(joinedload(AccountType.file_size))
         result = await db.execute(stmt)
         account_types = result.scalars().all()
         
         response = []
         
         for account_type in account_types:
            response.append(AccountTypeResponse(
                account_type_id=account_type.account_type_id,
                type_name=account_type.type_name,
                file_size=account_type.file_size.size 
            ))
         return response
     
    except Exception as e:
        print("="*50)
        print(f"ERROR TYPE: {type(e).__name__}")
        print(f"ERROR MESSAGE: {e}")
        print(traceback.format_exc())
        print("="*50)
        logging.error(f"Error fetching account types: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
@router.put("/account-types", tags=["Account Types"], response_model=AccountTypeResponse)
async def update_size_for_account_type(payload: AccountTypeUpdateSizeRequest, db: AsyncSession = Depends(get_db)):
    try:
        stmt = select(AccountType).options(joinedload(AccountType.file_size)).where(AccountType.account_type_id == payload.account_type_id)
        result = await db.execute(stmt)
        account_type = result.scalar_one_or_none()
        
        if not account_type:
            raise HTTPException(status_code=404, detail="Account type not found")
        
        # Update the file_size field
        account_type.file_size.size = payload.file_size
        await db.flush()
        await db.refresh(account_type)
        
        return AccountTypeResponse(
            account_type_id=account_type.account_type_id,
            type_name=account_type.type_name,
            file_size=account_type.file_size.size
        )
        
    except Exception as e:
        print("="*50)
        print(f"ERROR TYPE: {type(e).__name__}")
        print(f"ERROR MESSAGE: {e}")
        print(traceback.format_exc())
        print("="*50)
        logging.error(f"Error updating account type size: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
