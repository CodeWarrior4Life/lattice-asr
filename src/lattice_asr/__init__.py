"""lattice-asr — hardware-adaptive multilingual ASR."""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("lattice-asr")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from lattice_asr.telemetry import ListTelemetrySink, NullTelemetrySink
from lattice_asr.transcriber import Transcriber
from lattice_asr.types import (
    AsrCallRecord,
    EngineCapabilities,
    Segment,
    SpeakerSegment,
    TelemetrySink,
    TranscriptionResult,
)

__all__ = [
    "__version__",
    "Segment",
    "SpeakerSegment",
    "EngineCapabilities",
    "TranscriptionResult",
    "AsrCallRecord",
    "TelemetrySink",
    "NullTelemetrySink",
    "ListTelemetrySink",
    "Transcriber",
]

import asyncio
from typing import Optional

_singleton: Optional["Transcriber"] = None
_singleton_lock = asyncio.Lock()


def _reset_singleton() -> None:
    """Test-only — clear the lazy singleton."""
    global _singleton
    _singleton = None


async def transcribe(
    audio_pcm: bytes,
    sample_rate: int = 16000,
    language: str | None = None,
) -> "TranscriptionResult":
    """Module-level convenience using a process-singleton Transcriber. Spec §4.1."""
    global _singleton
    async with _singleton_lock:
        if _singleton is None:
            _singleton = Transcriber()
    return await _singleton.transcribe(audio_pcm, sample_rate, language=language)


__all__ += ["transcribe"]
