from pydantic import Field
from schemas.base import ORMBase

# --- Account Type Schemas ---
class AccountTypeResponse(ORMBase):
    account_type_id: int
    type_name: str
    file_size: int
    
class AccountTypeUpdateSizeRequest(ORMBase):
    account_type_id: int
    file_size: int
    
