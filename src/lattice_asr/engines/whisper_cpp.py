"""WhisperCppEngine — Apple Silicon CoreML multilingual stub. Implemented in W3.3. Spec §6.4."""

from __future__ import annotations

from collections.abc import AsyncIterator

from lattice_asr.engines.base import TranscriptionEngine
from lattice_asr.types import EngineCapabilities, TranscriptionResult


class WhisperCppEngine(TranscriptionEngine):
    def __init__(self, *, model: str = "whisper-large-v3-turbo-coreml"):
        self._model_name = model
        self.capabilities = EngineCapabilities(
            name="whisper.cpp",
            languages=frozenset({"en", "es", "fr", "de"}),
            streaming=True,
            requires_gpu=False,
            requires_apple_silicon=True,
            typical_rtfx=8.0,
        )

    async def transcribe(
        self,
        audio_pcm: bytes,
        sample_rate: int,
        language: str | None,
    ) -> TranscriptionResult:
        raise NotImplementedError("WhisperCppEngine implemented in W3.3")

    async def transcribe_streaming(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int,
        language: str | None,
    ) -> AsyncIterator[TranscriptionResult]:
        raise NotImplementedError("WhisperCppEngine implemented in W3.3")
        if False:
            yield  # type: ignore[unreachable]
