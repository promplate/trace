from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    langchain_api_key: str = ""
    langchain_project: str = ""
    langchain_endpoint: str = "https://api.smith.langchain.com"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def langsmith(self):
        return bool(self.langchain_api_key)

    @property
    def langfuse(self):
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    model_config = SettingsConfigDict(extra="allow", env_file=".env")


env = Settings()  # type: ignore
