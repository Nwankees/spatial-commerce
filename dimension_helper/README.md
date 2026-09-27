# Dimension helper

This optional local sidecar keeps Crawl4AI, Browser Use, and Playwright out of the
main backend environment. The backend calls it only after explicit structured
dimensions are insufficient.

```powershell
cd dimension_helper
.\setup.ps1
.\run.ps1
```

The backend uses `http://127.0.0.1:8020` by default. Crawl4AI renders the selected
page. Browser Use is the final fallback and talks to local Ollama using the
already-installed vision-capable `qwen3-vl:8b` (`BROWSER_USE_MODEL` can override
it); the main backend still uses `qwen3:4b-instruct` for typed dimension
extraction and verification. No cloud LLM is used.
