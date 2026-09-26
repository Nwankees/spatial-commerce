from __future__ import annotations

import base64
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.gemini_service import GeminiMalformedResponseError, GeminiUpstreamError
from app.main import app, get_analysis_service
from app.models import VisualProductAnalysis


class FakeAnalysisService:
    async def analyze(
        self,
        image_bytes: bytes,
        user_request: str | None,
    ) -> VisualProductAnalysis:
        assert image_bytes.startswith(b"\xff\xd8")
        assert user_request == "Find something similar but darker."
        return VisualProductAnalysis(
            objectDetected=True,
            category="chair",
            subcategory="accent chair",
            color="beige",
            materials=["fabric", "wood"],
            style=["modern", "minimalist"],
            shape="rounded",
            searchKeywords=["beige modern accent chair", "rounded fabric chair"],
            confidence=0.91,
        )


class FailingAnalysisService:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def analyze(self, image_bytes: bytes, user_request: str | None) -> VisualProductAnalysis:
        raise self.error


def jpeg_base64() -> str:
    buffer = BytesIO()
    Image.new("RGB", (4, 3), (210, 180, 145)).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_health_does_not_require_a_secret() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "geminiConfigured" in response.json()


def test_analyze_returns_typed_structured_result() -> None:
    app.dependency_overrides[get_analysis_service] = lambda: FakeAnalysisService()
    try:
        response = TestClient(app).post(
            "/api/v1/analyze",
            json={
                "imageBase64": jpeg_base64(),
                "mimeType": "image/jpeg",
                "rotationDegrees": 90,
                "userRequest": "Find something similar but darker.",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["objectDetected"] is True
    assert body["subcategory"] == "accent chair"
    assert body["confidence"] == 0.91


def test_analyze_rejects_invalid_image() -> None:
    app.dependency_overrides[get_analysis_service] = lambda: FakeAnalysisService()
    try:
        response = TestClient(app).post(
            "/api/v1/analyze",
            json={
                "imageBase64": base64.b64encode(b"not an image").decode("ascii"),
                "mimeType": "image/jpeg",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert "readable image" in response.json()["detail"]


def test_request_rejects_unknown_fields() -> None:
    response = TestClient(app).post(
        "/api/v1/analyze",
        json={
            "imageBase64": jpeg_base64(),
            "mimeType": "image/jpeg",
            "unexpected": "value",
        },
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (TimeoutError(), 504),
        (GeminiMalformedResponseError("bad schema"), 502),
        (GeminiUpstreamError("upstream failure"), 502),
    ],
)
def test_analysis_failures_are_retryable_http_errors(
    error: Exception,
    expected_status: int,
) -> None:
    app.dependency_overrides[get_analysis_service] = lambda: FailingAnalysisService(error)
    try:
        response = TestClient(app).post(
            "/api/v1/analyze",
            json={"imageBase64": jpeg_base64(), "mimeType": "image/jpeg"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == expected_status
    assert "retry" in response.json()["detail"].lower()
