from __future__ import annotations

import asyncio

from google import genai
from google.genai import types
from pydantic import ValidationError

from .models import VisualProductAnalysis
from .settings import Settings
from .vision_errors import VisionMalformedResponseError, VisionNotConfiguredError, VisionUnavailableError

# Milestone 5.5: Gemini is no longer on the active analysis path (see ollama_vision.py).
# This module is kept isolated and unused; its errors map onto the provider-neutral ones.


class GeminiNotConfiguredError(VisionNotConfiguredError):
    pass


class GeminiUpstreamError(VisionUnavailableError):
    pass


class GeminiMalformedResponseError(VisionMalformedResponseError):
    pass


class GeminiAnalysisService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def configured(self) -> bool:
        return bool(self._settings.gemini_api_key)

    async def analyze(
        self,
        image_bytes: bytes,
        user_request: str | None,
    ) -> VisualProductAnalysis:
        if not self._settings.gemini_api_key:
            raise GeminiNotConfiguredError("GEMINI_API_KEY is not configured.")

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._analyze_sync, image_bytes, user_request),
                timeout=self._settings.gemini_timeout_seconds,
            )
        except TimeoutError:
            raise
        except GeminiMalformedResponseError:
            raise
        except Exception as exc:
            raise GeminiUpstreamError("Gemini could not analyze the image.") from exc

    def _analyze_sync(
        self,
        image_bytes: bytes,
        user_request: str | None,
    ) -> VisualProductAnalysis:
        client = genai.Client(api_key=self._settings.gemini_api_key)
        prompt = _analysis_prompt(user_request)
        response = client.models.generate_content(
            model=self._settings.gemini_model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=_response_json_schema(),
                temperature=0.1,
            ),
        )

        try:
            if response.parsed is not None:
                if isinstance(response.parsed, VisualProductAnalysis):
                    result = response.parsed
                else:
                    result = VisualProductAnalysis.model_validate(response.parsed)
            elif response.text:
                result = VisualProductAnalysis.model_validate_json(response.text)
            else:
                raise GeminiMalformedResponseError("Gemini returned no structured result.")
            return VisualProductAnalysis.model_validate(result.model_dump())
        except (ValidationError, ValueError, TypeError) as exc:
            raise GeminiMalformedResponseError(
                "Gemini returned a response that did not match the product schema."
            ) from exc


def _analysis_prompt(user_request: str | None) -> str:
    context = (
        f"The user's optional shopping intent is: {user_request}"
        if user_request
        else "The user did not provide additional shopping intent."
    )
    return f"""
Identify the single principal physical product or object the user is intentionally pointing at.
Prefer a foreground, potentially shoppable object such as furniture, decor, clothing, electronics,
or a household item. Ignore people, walls, floors, and background clutter unless one of those is
clearly the intended product. Describe only visible or strongly supported attributes; do not invent
hidden materials, brands, prices, or model names. Generate useful shopping-search phrases from the
visual attributes. {context}

If no obvious product is visible, set objectDetected to false, confidence appropriately low,
category/subcategory/color/shape to null, materials/style/searchKeywords to empty arrays, and give
a short retry instruction in message. If a product is visible, set objectDetected to true and
message to null. Return only data matching the supplied schema.
""".strip()


def _response_json_schema() -> dict[str, object]:
    schema = VisualProductAnalysis.model_json_schema()
    # Gemini's JSON-schema mode accepts the typed properties but not Pydantic's
    # root-level additionalProperties flag. The response is still strictly
    # revalidated with the original Pydantic model before it leaves this service.
    schema.pop("additionalProperties", None)
    return schema
