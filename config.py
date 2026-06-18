from typing import Optional
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    mistral_api_key: Optional[SecretStr] = Field(default=None,validation_alias="API_KEY")
    model_config = SettingsConfigDict(
        env_file=".env",
        extra= "ignore"
    )

Config = Settings()