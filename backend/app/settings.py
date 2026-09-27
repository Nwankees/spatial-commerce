from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.8-flash"
    gemini_timeout_seconds: float = 45.0
    max_image_bytes: int = 8_000_000
    serpapi_api_key: str | None = None
    serpapi_timeout_seconds: float = 15.0
    serpapi_country: str = "us"
    serpapi_language: str = "en"
    product_cache_path: str = ".cache/product_search_cache.json"
    retailer_fetch_timeout_seconds: float = 8.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
