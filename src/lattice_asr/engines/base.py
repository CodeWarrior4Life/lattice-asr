"""TranscriptionEngine ABC — adapter base. Spec §4, §6."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from lattice_asr.types import EngineCapabilities, TranscriptionResult


class TranscriptionEngine(ABC):
    """Adapter for one ASR engine."""

    capabilities: EngineCapabilities

    @abstractmethod
    async def transcribe(
        self,
        audio_pcm: bytes,
        sample_rate: int,
        language: str | None,
    ) -> TranscriptionResult: ...

    @abstractmethod
    async def transcribe_streaming(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int,
        language: str | None,
    ) -> AsyncIterator[TranscriptionResult]:
        if False:  # makes this an async generator (subtype of AsyncIterator)
            yield  # type: ignore[unreachable]  # pragma: no cover

    async def warmup(self) -> None:
        pass
