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
