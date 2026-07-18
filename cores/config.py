# **********************************************************
#                   配置
# **********************************************************
from pydantic_settings import BaseSettings, SettingsConfigDict
from tools.constants import HOME_DIR
from functools import lru_cache

env_file = HOME_DIR / ".env"  # env文件


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=env_file,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # ====== llm openai ====== #
    OPENAI_MODEL: str
    OPENAI_API_KEY: str
    OPENAI_URL: str
    CONTEXT_WINDOW: int

    # ====== siliconflow ====== #
    SF_MODEL: str
    SF_API_KEY: str
    SF_EMBEDDING: str
    SF_URL: str


@lru_cache
def get_settings():
    return AgentSettings()
