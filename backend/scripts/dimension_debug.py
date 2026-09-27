"""Run the M6 page -> Qwen -> provenance -> normalized-dimensions pipeline.

Examples (from backend/):
  python scripts/dimension_debug.py --url https://www.walmart.com/ip/... --title "Chair" --retailer Walmart
  python scripts/dimension_debug.py --html ../tmp/m5v-0-walmart-0.html --url https://www.walmart.com/ip/x/17743069195 \
      --title "Chair" --retailer Walmart --space-width 0.80 --space-depth 0.80

This is intentionally verbose: it exposes exactly what was supplied to the
semantic model and exactly what deterministic validation accepted or rejected.
It never treats model output as a source of truth by itself.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Running a file under scripts/ puts that directory, not backend/, on sys.path.
# Add the service root so this command works exactly as documented from backend/.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.dimension_content import PragmaticDimensionResolver
from app.dimension_resolver import DimensionResolver
from app.dimension_semantic import OllamaSemanticDimensionExtractor
from app.main import get_dimension_resolver
from app.page_fetcher import FetchedPage, HttpPageFetcher
from app.product_models import ProductCandidate
from app.settings import get_settings


def dump(value: Any) -> str:
    if value is None:
        return "null"
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def heading(number: int, title: str) -> None:
    print(f"\n{'=' * 80}\n{number}. {title}\n{'=' * 80}")


class FileOrNetworkFetcher:
    def __init__(self, html_path: Path | None, timeout: float) -> None:
        self.html_path = html_path
        self.network = HttpPageFetcher(timeout_seconds=timeout)

    async def fetch(self, url: str) -> FetchedPage:
        if self.html_path:
            return FetchedPage(url, self.html_path.read_text(encoding="utf-8", errors="replace"))
        return await self.network.fetch(url)


def fit_report(result: Any, width: float | None, depth: float | None) -> dict[str, Any]:
    values = result.dimensionsMeters or []
    axis = result.axisMapping
    product_w = result.widthMeters
    product_d = result.depthMeters
    if axis and product_w is None and axis.widthIndex is not None and axis.widthIndex < len(values):
        product_w = values[axis.widthIndex]
    if axis and product_d is None and axis.depthIndex is not None and axis.depthIndex < len(values):
        product_d = values[axis.depthIndex]
    ar_eligible = len(values) == 3
    if product_w is None or product_d is None:
        return {
            "status": "UNKNOWN",
            "reason": (
                "retailer values are verified, but width/depth axis order is unresolved until mesh analysis"
                if ar_eligible
                else "verified product width and depth are required for Check Fit"
            ),
            "checkFitEligible": False,
            "arPreviewEligible": ar_eligible,
            "m6Eligible": ar_eligible,
        }
    if width is None or depth is None:
        return {
            "status": "NEEDS_SPACE_MEASUREMENT",
            "productWidthMeters": product_w,
            "productDepthMeters": product_d,
            "checkFitEligible": True,
            "arPreviewEligible": ar_eligible,
            "m6Eligible": True,
        }
    margin = 0.02
    orientations = [
        {"rotationDegrees": 0, "required": [product_w, product_d]},
        {"rotationDegrees": 90, "required": [product_d, product_w]},
    ]
    for item in orientations:
        rw, rd = item["required"]
        item["clearanceMeters"] = [round(width - rw, 5), round(depth - rd, 5)]
        item["fits"] = width + margin >= rw and depth + margin >= rd
    chosen = next((item for item in orientations if item["fits"]), max(
        orientations, key=lambda item: min(item["clearanceMeters"])))
    return {
        "status": "FITS" if chosen["fits"] else "DOES_NOT_FIT",
        "spaceMeters": [width, depth],
        "chosenOrientation": chosen,
        "allOrientations": orientations,
        "checkFitEligible": True,
        "arPreviewEligible": ar_eligible,
        "m6Eligible": True,
    }


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.html:
        # Keep deterministic saved-page debugging, but exercise the same two-stage
        # structured -> semantic wrapper as the live endpoint (without network fallbacks).
        extractor = OllamaSemanticDimensionExtractor(
            settings.ollama_base_url,
            settings.ollama_dimension_model,
            timeout_seconds=settings.ollama_dimension_timeout_seconds,
            num_ctx=settings.ollama_dimension_num_ctx,
            keep_alive=settings.ollama_keep_alive,
        )
        fast = DimensionResolver(
            None,
            FileOrNetworkFetcher(Path(args.html), settings.retailer_fetch_timeout_seconds),
            extractor=None,
            page_max_chars=settings.dimension_page_max_chars,
        )
        resolver = PragmaticDimensionResolver(
            fast,
            extractor,
            [],
            None,
            page_max_chars=settings.dimension_page_max_chars,
        )
    else:
        resolver = get_dimension_resolver(settings)
    candidate = ProductCandidate(
        id=args.product_id,
        provider=args.provider,
        providerProductId=args.product_id,
        title=args.title,
        price=args.price,
        retailer=args.retailer,
        productUrl=args.url,
        detailPageToken=args.detail_page_token,
    )

    heading(1, "Selected product / exact variant")
    print(dump(candidate))
    result = await resolver.resolve(candidate)
    trace = resolver.last_trace
    structured_trace = trace.get("structured_trace", {})
    print("Variant context:")
    print(dump(structured_trace.get("variant_context")))
    print("Pragmatic winner / timings:")
    print(dump({"winner": trace.get("winner"), "timings": trace.get("timings")}))

    attempts = trace.get("semantic_attempts", [])
    for index, attempt in enumerate(attempts, start=1):
        heading(2, f"Parsed structured page representation sent to Qwen (attempt {index})")
        print(dump({
            "provider": attempt.get("provider"),
            "sourceUrl": attempt.get("source_url"),
            "sourceName": attempt.get("source_label"),
        }))
        print(attempt.get("page_json") or dump(attempt["rep"].to_llm_dict(settings.dimension_page_max_chars)))

        heading(3, f"Raw Qwen result (attempt {index})")
        print(attempt.get("raw_llm") or "(no model response)")

        heading(4, f"Source field/path selected (attempt {index})")
        selected = None
        validation = attempt.get("validation")
        if validation and validation.source_ids:
            selected = {
                "sourceIds": validation.source_ids,
                "sourceType": validation.source_type,
                "sourcePath": validation.source_path,
                "rawText": validation.raw_text,
                "scope": validation.scope,
            }
        print(dump(selected or {"error": attempt.get("error")}))

        heading(5, f"Deterministic provenance validation (attempt {index})")
        print(dump(validation) if validation else dump({"error": attempt.get("error")}))

    if not attempts:
        fast_attempts = structured_trace.get("attempts", [])
        heading(2, "Structured fast-path attempts (Qwen was not needed)")
        print(dump([
            {
                "sourceName": attempt.name,
                "sourceUrl": attempt.url,
                "scope": attempt.scope,
                "fastPath": attempt.fast_path,
                "error": attempt.error,
            }
            for attempt in fast_attempts
        ]))

    heading(6, "Normalized dimensions")
    print(dump(result))

    heading(7, "Axis mapping")
    print(dump(result.axisMapping or {
        "status": "unresolved",
        "reason": "no verified values or source did not label the axes",
    }))

    heading(8, "Check fit / M6 eligibility")
    print(dump(fit_report(result, args.space_width, args.space_depth)))
    return 0 if result.status != "unavailable" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Exact selected retailer product URL")
    parser.add_argument("--html", help="Use a saved HTML file instead of fetching the URL")
    parser.add_argument("--title", required=True, help="Selected product title")
    parser.add_argument("--retailer", help="Retailer name")
    parser.add_argument("--product-id", default="debug:selected-product")
    parser.add_argument("--provider", default="debug", help="Candidate provider (use serpapi for a saved live candidate)")
    parser.add_argument("--detail-page-token", help="Optional provider detail-page token from the selected candidate")
    parser.add_argument("--price", type=float, default=0.0)
    parser.add_argument("--space-width", type=float, help="Measured available width in meters")
    parser.add_argument("--space-depth", type=float, help="Measured available depth in meters")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
