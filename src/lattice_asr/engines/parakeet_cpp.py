"""ParakeetCppEngine — Apple Silicon CoreML EN-only stub. Implemented in W3.1. Spec §6.1."""

from __future__ import annotations

from collections.abc import AsyncIterator

from lattice_asr.engines.base import TranscriptionEngine
from lattice_asr.types import EngineCapabilities, TranscriptionResult


class ParakeetCppEngine(TranscriptionEngine):
    def __init__(self, *, model: str = "parakeet-tdt-0.6b-v2-coreml"):
        self._model_name = model
        self.capabilities = EngineCapabilities(
            name="parakeet.cpp",
            languages=frozenset({"en"}),
            streaming=True,
            requires_gpu=False,
            requires_apple_silicon=True,
            typical_rtfx=15.0,
        )

    async def transcribe(
        self,
        audio_pcm: bytes,
        sample_rate: int,
        language: str | None,
    ) -> TranscriptionResult:
        raise NotImplementedError("ParakeetCppEngine implemented in W3.1")

    async def transcribe_streaming(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int,
        language: str | None,
    ) -> AsyncIterator[TranscriptionResult]:
        raise NotImplementedError("ParakeetCppEngine implemented in W3.1")
        if False:
            yield  # type: ignore[unreachable]
