from pydantic import BaseModel, Field
from core.enums import SSOLoginType

class SSOCodeRequest(BaseModel):
    code: str
    type: SSOLoginType = Field(
        ...,
        description="Environment type"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"code": "abc123", "type": "fastapi"},
                {"code": "abc123", "type": "vercel"},
                {"code": "abc123", "type": "local"},
            ]
        }
    }
