from pydantic import BaseModel


class SSOCodeRequest(BaseModel):
    code: str
