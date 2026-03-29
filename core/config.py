import os
import pathlib

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    database_url: str
    secret_key: str = os.getenv("SECRET_KEY")
    access_token_expire_minutes: int = 720
    upload_root: pathlib.Path = pathlib.Path("./file_storage/uploads")
    TRASH_DIR: pathlib.Path = pathlib.Path("./file_storage/trash")
    HASH_SALT: str = os.getenv("HASH_SALT")
    MIN_LENGTH: int = int(os.getenv("MIN_LENGTH", 8))
    SSO_CLIENT_ID: str = os.getenv("SSO_CLIENT_ID")
    SSO_CLIENT_SECRET: str = os.getenv("SSO_CLIENT_SECRET")
    SSO_REDIRECT_URI: str = os.getenv("SSO_REDIRECT_URI")
    SSO_TOKEN_URL: str = os.getenv("SSO_TOKEN_URL")
    SSO_USERINFO_URL: str = os.getenv("SSO_USERINFO_URL")
    
    # === OpenRouter ===
    use_openrouter: bool = os.getenv("USE_OPENROUTER", "false").lower() == "true"
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "mistralai/ministral-3b-2512")
    openrouter_context_window: int = int(os.getenv("OPENROUTER_CONTEXT_WINDOW", "262144"))
    openrouter_referer: str = os.getenv("OPENROUTER_REFERER", "http://localhost")
    openrouter_app_title: str = os.getenv("OPENROUTER_APP_TITLE", "RAG-App")
    
    # === Google Gemini ===
    use_gemini: bool = os.getenv("USE_GEMINI", "false").lower() == "true"
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash")
    gemini_context_window: int = int(os.getenv("GEMINI_CONTEXT_WINDOW", "1048576"))
    class Config:
        env_file = ".env"

settings = Settings()