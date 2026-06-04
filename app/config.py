from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    hf_api_token: str = ""
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"


settings = Settings()