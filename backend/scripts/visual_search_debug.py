r"""Live two-photo trace for visual + text product retrieval.

Run from backend/:
    .venv\Scripts\python scripts\visual_search_debug.py chair.jpg mouse.jpg

The script performs real Ollama and SerpApi calls and deliberately prints no
secret values. Unit tests never invoke it.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Running a file under scripts/ puts that directory, not backend/, on sys.path.
# Add the service root so the documented command works from backend/.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.image_processing import prepare_image, prepare_lens_upload
from app.models import AnalyzeProductRequest
from app.ollama_vision import OllamaVisionService
from app.product_search_service import ProductSearchService
from app.query_builder import QueryPlanner
from app.serpapi_provider import SerpApiProductSearchProvider
from app.settings import Settings
from app.visual_reranker import OllamaCandidateVisualReranker
from app.visual_search import SerpApiGoogleLensProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def dump(value) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


async def inspect_photo(path: Path, settings: Settings) -> None:
    print(f"\n{'=' * 80}\nPHOTO: {path}\n{'=' * 80}")
    raw = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    request = AnalyzeProductRequest(
        imageBase64=base64.b64encode(raw).decode("ascii"),
        mimeType=mime,
    )
    image = prepare_image(request, settings.max_image_bytes)

    vision = OllamaVisionService(
        settings.ollama_base_url,
        settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        keep_alive=settings.ollama_keep_alive,
        min_confidence=settings.vision_min_confidence,
    )
    analysis_started = time.perf_counter()
    analysis = await vision.analyze(image, None)
    analysis_ms = round((time.perf_counter() - analysis_started) * 1000)
    print("\n1. Qwen visual analysis")
    dump(analysis.model_dump(mode="json"))

    planner = QueryPlanner(max_queries=settings.product_search_max_queries)
    queries = planner.plan(analysis)
    print("\n2. Text search queries")
    dump([query.text for query in queries])

    text_provider = SerpApiProductSearchProvider(
        settings.serpapi_api_key,
        timeout_seconds=settings.serpapi_timeout_seconds,
        country=settings.serpapi_country,
        language=settings.serpapi_language,
    )
    modes = tuple(mode.strip() for mode in settings.lens_modes.split(",") if mode.strip())
    lens_provider = SerpApiGoogleLensProvider(
        settings.serpapi_api_key,
        timeout_seconds=settings.serpapi_timeout_seconds,
        country=settings.serpapi_country,
        language=settings.serpapi_language,
        modes=modes,
        limit_per_mode=settings.product_search_results_per_query,
    )
    text_started = time.perf_counter()
    text_task = asyncio.gather(
        *(text_provider.search(query, settings.product_search_results_per_query) for query in queries),
        return_exceptions=True,
    )
    lens_task = lens_provider.search(
        prepare_lens_upload(image),
        analysis.subcategory or analysis.category,
        analysis.searchQueries[0] if analysis.searchQueries else analysis.subcategory,
    )
    text_outcomes, lens_outcome = await asyncio.gather(text_task, lens_task, return_exceptions=True)
    text_ms = round((time.perf_counter() - text_started) * 1000)
    text_results = [
        (query, outcome)
        for query, outcome in zip(queries, text_outcomes)
        if isinstance(outcome, list)
    ]

    if isinstance(lens_outcome, BaseException):
        print(f"\n3-4. Lens search failed: {type(lens_outcome).__name__}: {lens_outcome}")
        visual = None
    else:
        visual = lens_outcome
        print("\n3. Lens upload/search result")
        dump({
            "uploadMs": visual.upload_ms,
            "searchMs": visual.search_ms,
            "modeErrors": visual.mode_errors,
        })
        print("\n4. Lens products")
        dump([item.product.model_dump(mode="json") for item in visual.candidates if item.mode == "products"])
        print("\n5. Lens visual matches")
        dump([item.product.model_dump(mode="json") for item in visual.candidates if item.mode == "visual_matches"])
        print("\n6. Lens exact matches")
        dump([item.product.model_dump(mode="json") for item in visual.candidates if item.mode == "exact_matches"])

    service = ProductSearchService(text_provider, None, planner=planner)
    merge_started = time.perf_counter()
    merged = service._merge(text_results, visual)  # Debug script intentionally exposes pipeline stages.
    ranked = service._reranker.rerank(analysis, list(merged.values()), queries[0].text)
    merge_ms = round((time.perf_counter() - merge_started) * 1000)
    generated_count = sum(len(outcome) for _, outcome in text_results)
    if visual:
        generated_count += len(visual.candidates)
    print("\n7. Merge/dedupe summary")
    dump({
        "generatedCandidates": generated_count,
        "mergedCandidates": len(merged),
        "duplicateRemovals": max(0, generated_count - len(merged)),
    })
    print("\n8. Merged/deduped candidates")
    dump([{
        "id": item.candidate.id,
        "title": item.candidate.title,
        "sources": item.retrieval_sources,
        "textRank": item.text_rank,
        "visualRank": item.visual_rank,
        "deterministicScore": item.score,
        "scoreBreakdown": item.breakdown,
    } for item in ranked])

    local_ms = None
    if settings.visual_rerank_enabled and ranked:
        rerank_model = getattr(settings, "visual_rerank_model", settings.ollama_model)
        try:
            local = await OllamaCandidateVisualReranker(
                settings.ollama_base_url,
                rerank_model,
                timeout_seconds=settings.visual_rerank_timeout_seconds,
                max_candidates=settings.visual_rerank_max_candidates,
                num_ctx=settings.visual_rerank_num_ctx,
                keep_alive=settings.ollama_keep_alive,
            ).rerank(image, [item.candidate for item in ranked])
        except Exception as exc:
            print("\n9. Local visual reranking failed open")
            dump({"model": rerank_model, "errorType": type(exc).__name__, "reason": str(exc)})
        else:
            local_ms = local.duration_ms
            print("\n9. Local visual reranking scores")
            dump({
                "model": rerank_model,
                "scores": {
                    candidate_id: score.model_dump(mode="json")
                    for candidate_id, score in local.scores.items()
                },
            })
            for item in ranked:
                score = local.scores.get(item.candidate.id)
                if score:
                    item.same_product_probability = score.sameProductProbability
                    item.visual_similarity = score.visualSimilarity
                    item.visual_category_match = score.categoryMatch
            ranked = service._reranker.rerank(analysis, ranked, queries[0].text)

    final = [service._finalize(item) for item in ranked[:5]]
    print("\n10. Final top 5")
    dump([product.model_dump(mode="json") for product in final])
    print("\nLatency breakdown (ms)")
    dump({
        "analysis": analysis_ms,
        "textAndLensWallClock": text_ms,
        "lensUpload": visual.upload_ms if visual else None,
        "lensSearch": visual.search_ms if visual else None,
        "mergeAndDeterministicRerank": merge_ms,
        "localVisualRerank": local_ms,
        "total": round((time.perf_counter() - analysis_started) * 1000),
    })


async def main(paths: list[Path]) -> None:
    settings = Settings()
    if not settings.serpapi_api_key:
        raise SystemExit("SERPAPI_API_KEY is missing from backend/.env")
    for path in paths:
        await inspect_photo(path, settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace visual + text retrieval for two real photos.")
    parser.add_argument("photos", nargs=2, type=Path, help="Two JPEG/PNG product photos")
    args = parser.parse_args()
    asyncio.run(main(args.photos))
