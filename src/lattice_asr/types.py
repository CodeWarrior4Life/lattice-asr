"""Public API dataclasses — frozen, hashable. Spec §4."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class Segment:
    text: str
    start_ms: int
    end_ms: int
    confidence: float


@dataclass(frozen=True)
class SpeakerSegment:
    label: str
    start_ms: int
    end_ms: int
    text: str
    voice_print_id: str | None = None


@dataclass(frozen=True)
class EngineCapabilities:
    name: str
    languages: frozenset[str]
    streaming: bool
    requires_gpu: bool
    requires_apple_silicon: bool
    typical_rtfx: float


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    confidence: float
    engine_name: str
    segments: tuple[Segment, ...] = field(default_factory=tuple)
    speaker_segments: tuple[SpeakerSegment, ...] = field(default_factory=tuple)
    audio_duration_ms: int = 0
    duration_ms: int = 0
    # Non-None means THE LANGUAGE OF THIS TRANSCRIPT IS NOT TRUSTWORTHY, with a
    # human-readable reason. Added 2026-09-01: the defect that cost 80% of a
    # client call was not a missing detector, it was that an unreliable language
    # decision looked exactly like a reliable one. A consumer surfacing a
    # transcript should surface this alongside it.
    language_warning: str | None = None
    #: WHERE `language` came from, because the grades are not equivalent:
    #:   "requested"       -- the caller pinned it; no detection happened
    #:   "engine"          -- the engine reported its own detection
    #:   "audio-lid"       -- a detector identified it from the AUDIO
    #:   "transcript-text" -- inferred from the TEXT the engine produced; weaker,
    #:                        since it describes what was written, not what was
    #:                        spoken, and those diverge exactly when the engine
    #:                        mistranslates
    #:   "undetermined"    -- genuinely unknown; `language` is "und"
    language_source: str = "engine"


@dataclass(frozen=True)
class AsrCallRecord:
    engine_name: str
    language_detected: str
    language_requested: str | None
    audio_duration_ms: int
    transcription_duration_ms: int
    diarized: bool
    speaker_count: int | None
    tenant_id: str | None
    timestamp_utc: datetime
    # Detector probability for `language_detected`, or None if nothing detected.
    language_confidence: float | None = None
    # Mirrors TranscriptionResult.language_warning so a telemetry sink can alert
    # on silent-fallback rates instead of waiting for someone to read a bad
    # transcript in a language they happen to speak.
    language_warning: str | None = None
    language_source: str = "engine"


class TelemetrySink(Protocol):
    def record(self, call: AsrCallRecord) -> None: ...
