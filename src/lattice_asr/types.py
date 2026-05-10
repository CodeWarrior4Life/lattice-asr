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


class TelemetrySink(Protocol):
    def record(self, call: AsrCallRecord) -> None: ...
