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
    # The 8B model is the live-demo default: it stays responsive on the event
    # laptop while preserving the same schema-constrained vision contract.
    ollama_model: str = "qwen3-vl:8b"
    ollama_timeout_seconds: float = 180.0
    ollama_keep_alive: str = "15m"
    vision_min_confidence: float = 0.2
    # Separate small text model that reads the parsed product page and points at the
    # stated dimensions (validated deterministically afterwards).
    ollama_dimension_model: str = "qwen3:4b-instruct"
    ollama_dimension_timeout_seconds: float = 90.0
    ollama_dimension_num_ctx: int = 8192
    dimension_llm_enabled: bool = True
    dimension_page_max_chars: int = 12000
    # Optional isolated helper hosting Crawl4AI + Browser Use. If it is not
    # running, exact SerpApi web search still works and failures degrade safely.
    dimension_helper_url: str = "http://127.0.0.1:8020"
    dimension_crawl_enabled: bool = True
    dimension_web_search_enabled: bool = True
    dimension_browser_use_enabled: bool = True
    dimension_crawl_timeout_seconds: float = 75.0
    dimension_browser_timeout_seconds: float = 240.0
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
    lens_search_enabled: bool = True
    lens_modes: str = "products,visual_matches,exact_matches"
    visual_rerank_enabled: bool = True
    visual_rerank_model: str = "qwen3-vl:8b"
    # Compare only the leading products, one candidate per local vision call.
    # This avoids long multi-image reasoning while keeping demo latency bounded.
    visual_rerank_max_candidates: int = 3
    visual_rerank_timeout_seconds: float = 180.0
    visual_rerank_num_ctx: int = 16384
    product_cache_path: str = ".cache/product_search_cache.json"
    retailer_fetch_timeout_seconds: float = 8.0
    # Milestone 6: real-product AR preview (local reconstruction sidecar).
    reconstruction_service_url: str = "http://127.0.0.1:8010"
    reconstruction_timeout_seconds: float = 300.0
    ar_asset_cache_dir: str = ".cache/ar-assets"


@lru_cache
def get_settings() -> Settings:
    return Settings()
