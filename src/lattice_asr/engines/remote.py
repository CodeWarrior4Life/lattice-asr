"""RemoteEngine — HTTP forwarding. Spec §6.5."""

from __future__ import annotations

import asyncio
import base64
import time
from typing import AsyncIterator

import httpx

from lattice_asr.engines.base import TranscriptionEngine
from lattice_asr.types import (
    EngineCapabilities,
    Segment,
    SpeakerSegment,
    TranscriptionResult,
)


_API_VERSION = "v1"
_RETRIES = 3
_BACKOFF_BASE = 0.25


class RemoteEngineError(Exception):
    pass


class RemoteEngine(TranscriptionEngine):
    def __init__(
        self,
        *,
        url: str,
        api_key: str | None = None,
        timeout: float = 10.0,
        _transport: httpx.MockTransport | None = None,
    ):
        self._url = url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._transport = _transport
        self.capabilities = EngineCapabilities(
            name="remote",
            languages=frozenset({"en"}),
            streaming=True,
            requires_gpu=False,
            requires_apple_silicon=False,
            typical_rtfx=20.0,
        )

    def _client(self) -> httpx.AsyncClient:
        kwargs: dict = {"timeout": self._timeout}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.AsyncClient(**kwargs)

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    async def transcribe(
        self, audio_pcm: bytes, sample_rate: int, language: str | None
    ) -> TranscriptionResult:
        body = {
            "audio_b64": base64.b64encode(audio_pcm).decode("ascii"),
            "sample_rate": sample_rate,
            "language": language,
            "diarize": False,
        }
        t0 = time.monotonic()
        last_err: Exception | None = None
        async with self._client() as client:
            for attempt in range(_RETRIES):
                try:
                    resp = await client.post(
                        f"{self._url}/v1/transcribe",
                        json=body,
                        headers=self._headers(),
                    )
                except httpx.HTTPError as exc:
                    last_err = exc
                    await asyncio.sleep(_BACKOFF_BASE * (2**attempt))
                    continue
                if 500 <= resp.status_code < 600:
                    last_err = RemoteEngineError(f"server {resp.status_code}: {resp.text[:200]}")
                    await asyncio.sleep(_BACKOFF_BASE * (2**attempt))
                    continue
                if resp.status_code >= 400:
                    raise RemoteEngineError(f"client error {resp.status_code}: {resp.text[:200]}")
                api_version = resp.headers.get("X-Lattice-Asr-Api-Version", _API_VERSION)
                if api_version != _API_VERSION:
                    raise RemoteEngineError(
                        f"wire-protocol version mismatch: client={_API_VERSION} server={api_version}"
                    )
                data = resp.json()
                segs = tuple(Segment(**s) for s in data.get("segments", []))
                spk = tuple(SpeakerSegment(**s) for s in data.get("speaker_segments", []))
                return TranscriptionResult(
                    text=data["text"],
                    language=data["language"],
                    confidence=data["confidence"],
                    engine_name=data["engine_name"],
                    segments=segs,
                    speaker_segments=spk,
                    audio_duration_ms=data.get("audio_duration_ms", 0),
                    duration_ms=data.get("duration_ms", int((time.monotonic() - t0) * 1000)),
                )
        raise RemoteEngineError(f"exhausted {_RETRIES} retries: {last_err}")

    async def transcribe_streaming(
        self, audio_chunks: AsyncIterator[bytes], sample_rate: int, language: str | None
    ) -> AsyncIterator[TranscriptionResult]:
        buf = bytearray()
        async for chunk in audio_chunks:
            buf.extend(chunk)
            if len(buf) >= sample_rate * 2:
                yield await self.transcribe(bytes(buf), sample_rate, language)
                buf.clear()
        if buf:
            yield await self.transcribe(bytes(buf), sample_rate, language)
