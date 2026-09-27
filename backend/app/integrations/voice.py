from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx

from .contracts import SpeechAudio, VoiceProvider

logger = logging.getLogger("app.integrations.voice")


class VoiceUnavailableError(RuntimeError):
    pass


class TextOnlyVoiceProvider:
    name = "text_only"
    remote = False

    async def synthesize(self, text: str) -> SpeechAudio:
        raise VoiceUnavailableError("Voice is not configured; use the text response.")


class ElevenLabsVoiceProvider:
    name = "elevenlabs"
    remote = True

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        *,
        model_id: str = "eleven_flash_v2_5",
        output_format: str = "mp3_44100_128",
        base_url: str = "https://api.elevenlabs.io",
        timeout_seconds: float = 20.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key or not voice_id:
            raise ValueError("ElevenLabs API key and voice ID are required")
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = output_format
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._client = client

    async def _request(self, url: str, **kwargs: Any) -> httpx.Response:
        if self._client is not None:
            response = await self._client.post(url, **kwargs)
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, **kwargs)
        response.raise_for_status()
        return response

    async def synthesize(self, text: str) -> SpeechAudio:
        url = f"{self._base_url}/v1/text-to-speech/{quote(self._voice_id, safe='')}"
        response = await self._request(
            url,
            headers={"xi-api-key": self._api_key, "Content-Type": "application/json"},
            params={"output_format": self._output_format},
            json={"text": text, "model_id": self._model_id},
        )
        if not response.content:
            raise VoiceUnavailableError("ElevenLabs returned empty audio")
        return SpeechAudio(
            data=response.content,
            content_type=response.headers.get("content-type", "audio/mpeg").split(";", 1)[0],
        )


class FailOpenVoiceProvider:
    def __init__(self, primary: VoiceProvider | None) -> None:
        self._primary = primary
        self.name = primary.name if primary else "text_only"
        self.remote = bool(primary and primary.remote)
        self.last_error: str | None = None

    async def synthesize(self, text: str) -> SpeechAudio | None:
        self.last_error = None
        if self._primary is None:
            self.last_error = "Voice is not configured."
            return None
        try:
            return await self._primary.synthesize(text)
        except Exception as exc:
            logger.warning("ElevenLabs request failed open: %s", type(exc).__name__)
            self.last_error = "Voice is temporarily unavailable."
            return None
