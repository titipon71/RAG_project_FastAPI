import enum


class RoleUser(str, enum.Enum):
    user = "user"
    admin = "admin"
    special = "special"

class RoleChannel(str, enum.Enum):
    public = "public"
    private = "private"
    pending = "pending"

class Theme(str, enum.Enum):
    light = "light"
    dark = "dark"
    
class ModerationDecision(str, enum.Enum):
    approved = "approved"
    rejected = "rejected"
    
class SSOLoginType(str, enum.Enum):
    fastapi = "fastapi"
    vercel = "vercel"
    local = "local"
    local2 = "local2"
    
class OCRTool(str, enum.Enum):
    easyocr = "easyocr"
    paddleocr = "paddleocr-vl"
    lightonocr = "lightonocr"