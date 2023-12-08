from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    langchain_api_key: str
    langchain_project: str
    langchain_endpoint: str = "https://api.smith.langchain.com"

    model_config = SettingsConfigDict(extra="allow", env_file=".env")


env = Settings()
