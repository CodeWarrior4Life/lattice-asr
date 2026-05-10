"""Transcriber — engine selection + transcribe routing. Spec §4, §5."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from lattice_asr.config import LatticeAsrConfig
from lattice_asr.engines.base import TranscriptionEngine
from lattice_asr.engines.faster_whisper import FasterWhisperEngine
from lattice_asr.hardware import HardwareProfile, detect_hardware
from lattice_asr.lid import SileroLid
from lattice_asr.telemetry import NullTelemetrySink
from lattice_asr.types import (
    AsrCallRecord,
    TelemetrySink,
    TranscriptionResult,
)


def _build_engine_registry(
    hw: HardwareProfile, force: str | None
) -> dict[str, TranscriptionEngine]:
    """Build {language_route: engine}. Spec §5.

    Routes: "en" (English-optimized) and "multi" (multilingual fallback).
    Lazy-imports per-platform engines to avoid pulling heavy deps unless needed.
    """
    if force:
        if force == "faster-whisper":
            engine = FasterWhisperEngine(
                model="distil-large-v3",
                device="cuda" if hw.nvidia_cuda else "cpu",
                compute_type="float16" if hw.nvidia_cuda else "int8",
            )
            return {"en": engine, "multi": engine}
        if force.startswith("remote:"):
            from lattice_asr.engines.remote import RemoteEngine

            url = force.split(":", 1)[1]
            if not url:
                raise ValueError("force_engine='remote:' requires a URL, got empty")
            engine = RemoteEngine(url=url)
            return {"en": engine, "multi": engine}
        if force == "parakeet.cpp":
            from lattice_asr.engines.parakeet_cpp import ParakeetCppEngine

            engine = ParakeetCppEngine()
            return {"en": engine, "multi": engine}
        if force == "parakeet-tdt":
            from lattice_asr.engines.parakeet_tdt import ParakeetTdtEngine

            engine = ParakeetTdtEngine()
            return {"en": engine, "multi": engine}
        if force == "whisper.cpp":
            from lattice_asr.engines.whisper_cpp import WhisperCppEngine

            engine = WhisperCppEngine()
            return {"en": engine, "multi": engine}
        raise ValueError(f"unknown force_engine: {force}")

    if hw.apple_silicon:
        from lattice_asr.engines.parakeet_cpp import ParakeetCppEngine
        from lattice_asr.engines.whisper_cpp import WhisperCppEngine

        return {
            "en": ParakeetCppEngine(),
            "multi": WhisperCppEngine(),
        }

    if hw.nvidia_cuda and hw.cuda_capability is not None and hw.cuda_capability >= (7, 0):
        from lattice_asr.engines.parakeet_tdt import ParakeetTdtEngine

        return {
            "en": ParakeetTdtEngine(),
            "multi": FasterWhisperEngine(
                model="distil-large-v3", device="cuda", compute_type="float16"
            ),
        }

    cpu_engine = FasterWhisperEngine(model="distil-large-v3", device="cpu", compute_type="int8")
    return {"en": cpu_engine, "multi": cpu_engine}


class Transcriber:
    """Hardware-adaptive ASR with optional diarization. Spec §4."""

    def __init__(
        self,
        *,
        default_language: str = "en",
        config: LatticeAsrConfig | None = None,
        telemetry_sink: TelemetrySink | None = None,
        force_engine: str | None = None,
        enable_diarization: bool = False,
    ):
        if enable_diarization:
            raise NotImplementedError(
                "Diarization is not available in v0.1; it lands in W5. "
                "Construct Transcriber with enable_diarization=False (default) and re-enable "
                "after upgrading to a lattice-asr release that exposes a diarizer."
            )
        self._default_language = default_language
        self._config = config or LatticeAsrConfig()
        self._telemetry = telemetry_sink or NullTelemetrySink()
        self._enable_diarization = enable_diarization
        self._hardware = detect_hardware()
        self._engines = _build_engine_registry(
            self._hardware, force_engine or self._config.hardware_force
        )
        self._lid = SileroLid()
        self._diarizer = None  # loaded lazily in W5

    @property
    def hardware(self) -> HardwareProfile:
        return self._hardware

    @property
    def loaded_engines(self) -> dict[str, TranscriptionEngine]:
        return dict(self._engines)

    async def warmup(self) -> None:
        await asyncio.gather(*(e.warmup() for e in set(self._engines.values())))
        if self._config.lid.enabled:
            await self._lid.warmup()

    async def transcribe(
        self,
        audio_pcm: bytes,
        sample_rate: int = 16000,
        *,
        language: str | None = None,
        diarize: bool = False,
        tenant_id: str | None = None,
    ) -> TranscriptionResult:
        if diarize and not self._enable_diarization:
            raise ValueError("diarize=True requires enable_diarization=True at __init__")

        requested = language
        if language is None and self._config.lid.enabled:
            # SileroLid internally trims to LID_AUDIO_SECONDS (spec §8); no external slice needed.
            lid_result = await self._lid.detect(audio_pcm, sample_rate)
            if lid_result.confidence >= self._config.lid.confidence_threshold:
                language = lid_result.language
            else:
                language = self._default_language
        elif language is None:
            language = self._default_language

        route = "en" if language == "en" else "multi"
        engine = self._engines[route]
        result = await engine.transcribe(audio_pcm, sample_rate, language)

        speaker_count: int | None = None
        if diarize and self._diarizer is not None:
            from dataclasses import replace
            from lattice_asr.diarize import merge_segments_with_text

            speaker_segments = await self._diarizer.diarize(audio_pcm, sample_rate)
            speaker_segments = merge_segments_with_text(speaker_segments, result.segments)
            result = replace(result, speaker_segments=tuple(speaker_segments))
            speaker_count = len({s.label for s in speaker_segments})

        self._telemetry.record(
            AsrCallRecord(
                engine_name=result.engine_name,
                language_detected=result.language,
                language_requested=requested,
                audio_duration_ms=result.audio_duration_ms,
                transcription_duration_ms=result.duration_ms,
                diarized=diarize,
                speaker_count=speaker_count,
                tenant_id=tenant_id,
                timestamp_utc=datetime.now(UTC),
            )
        )
        return result

    async def transcribe_streaming(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int = 16000,
        *,
        language: str | None = None,
        diarize: bool = False,
        tenant_id: str | None = None,
    ) -> AsyncIterator[TranscriptionResult]:
        if diarize and not self._enable_diarization:
            raise ValueError("diarize=True requires enable_diarization=True at __init__")
        # For v0.1, route by static language hint; LID per-chunk is deferred
        lang = language or self._default_language
        route = "en" if lang == "en" else "multi"
        engine = self._engines[route]
        async for partial in engine.transcribe_streaming(audio_chunks, sample_rate, lang):
            self._telemetry.record(
                AsrCallRecord(
                    engine_name=partial.engine_name,
                    language_detected=partial.language,
                    language_requested=language,
                    audio_duration_ms=partial.audio_duration_ms,
                    transcription_duration_ms=partial.duration_ms,
                    diarized=False,  # v0.1: streaming + diarize not combined
                    speaker_count=None,
                    tenant_id=tenant_id,
                    timestamp_utc=datetime.now(UTC),
                )
            )
            yield partial
