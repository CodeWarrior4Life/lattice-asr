"""lattice-asr — hardware-adaptive multilingual ASR."""

try:
    from importlib.metadata import version

    __version__ = version("lattice-asr")
except Exception:
    __version__ = "0.0.0+unknown"

from lattice_asr.types import (
    Segment,
    SpeakerSegment,
    EngineCapabilities,
    TranscriptionResult,
    AsrCallRecord,
    TelemetrySink,
)
from lattice_asr.telemetry import NullTelemetrySink, ListTelemetrySink

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
]
