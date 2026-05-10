"""ParakeetTdtEngine — NVIDIA NeMo Parakeet-TDT EN-only stub. Implemented in W3.2. Spec §6.2."""

from __future__ import annotations

from collections.abc import AsyncIterator

from lattice_asr.engines.base import TranscriptionEngine
from lattice_asr.types import EngineCapabilities, TranscriptionResult


class ParakeetTdtEngine(TranscriptionEngine):
    def __init__(self, *, model: str = "parakeet-tdt-1.1b-v3"):
        self._model_name = model
        self.capabilities = EngineCapabilities(
            name="parakeet-tdt",
            languages=frozenset({"en"}),
            streaming=True,
            requires_gpu=True,
            requires_apple_silicon=False,
            typical_rtfx=50.0,
        )

    async def transcribe(
        self,
        audio_pcm: bytes,
        sample_rate: int,
        language: str | None,
    ) -> TranscriptionResult:
        raise NotImplementedError("ParakeetTdtEngine implemented in W3.2")

    async def transcribe_streaming(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int,
        language: str | None,
    ) -> AsyncIterator[TranscriptionResult]:
        raise NotImplementedError("ParakeetTdtEngine implemented in W3.2")
        if False:
            yield  # type: ignore[unreachable]
