import enum


class RoleUser(str, enum.Enum):
    user = "user"
    admin = "admin"

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