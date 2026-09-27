from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Active local vision model (Milestone 5.5).
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3-vl:30b"
    ollama_timeout_seconds: float = 180.0
    ollama_keep_alive: str = "15m"
    vision_min_confidence: float = 0.2
    # Separate small text model that only *extracts* dimensions from retailer text.
    ollama_dimension_model: str = "qwen3:4b-instruct"
    ollama_dimension_timeout_seconds: float = 60.0
    dimension_llm_enabled: bool = True
    dimension_evidence_max_chars: int = 6000
    # Multi-query retrieval.
    product_search_max_queries: int = 3
    product_search_results_per_query: int = 10
    # Legacy Gemini settings (unused by the active path).
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
