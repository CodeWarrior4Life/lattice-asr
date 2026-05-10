"""RemoteEngine — HTTP/2 client to lattice-asr-server stub. Implemented in W4.1. Spec §6.5."""

from __future__ import annotations

from collections.abc import AsyncIterator

from lattice_asr.engines.base import TranscriptionEngine
from lattice_asr.types import EngineCapabilities, TranscriptionResult


class RemoteEngine(TranscriptionEngine):
    def __init__(
        self,
        *,
        url: str,
        api_key: str | None = None,
        timeout: float = 10.0,
    ):
        self._url = url
        self._api_key = api_key
        self._timeout = timeout
        self.capabilities = EngineCapabilities(
            name="remote",
            languages=frozenset({"en"}),  # set at handshake (W4.1)
            streaming=True,
            requires_gpu=False,
            requires_apple_silicon=False,
            typical_rtfx=20.0,
        )

    async def transcribe(
        self,
        audio_pcm: bytes,
        sample_rate: int,
        language: str | None,
    ) -> TranscriptionResult:
        raise NotImplementedError("RemoteEngine implemented in W4.1")

    async def transcribe_streaming(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int,
        language: str | None,
    ) -> AsyncIterator[TranscriptionResult]:
        raise NotImplementedError("RemoteEngine implemented in W4.1")
        if False:
            yield  # type: ignore[unreachable]
