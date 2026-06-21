from typing import Optional
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Let Pydantic automatically look for MISTRAL_API_KEY
    mistral_api_key: SecretStr
    langsmith_api_key: SecretStr
    langsmith_tracing: bool
    langsmith_project: str
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

Config = Settings() # type: ignore