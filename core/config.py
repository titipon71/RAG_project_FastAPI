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
    class Config:
        env_file = ".env"

settings = Settings()