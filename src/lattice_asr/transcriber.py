"""Transcriber — engine selection + transcribe routing. Spec §4, §5."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from lattice_asr.config import LatticeAsrConfig
from lattice_asr.engines.base import EngineUnavailableError, TranscriptionEngine
from lattice_asr.engines.faster_whisper import FasterWhisperEngine
from lattice_asr.hardware import HardwareProfile, detect_hardware
from lattice_asr.lid import UNDETERMINED, LidResult
from lattice_asr.telemetry import NullTelemetrySink
from lattice_asr.types import (
    AsrCallRecord,
    TelemetrySink,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)


# The multilingual route's model, per device class. NOT distil-large-v3, which is
# English-only and was the previous default on every route: MEASURED 2026-09-01 it
# reports en@0.9463 on Spanish audio and translates instead of transcribing.
# large-v3 on GPU (best multilingual quality, and the only option here that covers
# ja/zh/ar properly); medium on CPU, where large-v3 int8 is impractically slow and
# medium already measured es@0.9468.
MULTILINGUAL_MODEL_CUDA = "large-v3"
MULTILINGUAL_MODEL_CPU = "medium"


def _build_engine_registry(
    hw: HardwareProfile, force: str | None
) -> dict[str, TranscriptionEngine]:
    """Build {language_route: engine}. Spec §5.

    Routes: "en" (English-optimized) and "multi" (multilingual fallback). The keys
    are historical: selection is by each engine's declared `capabilities.languages`
    and availability, not by these names (see `_select_engine`).

    Lazy-imports per-platform engines to avoid pulling heavy deps unless needed.
    """
    if force:
        if force == "faster-whisper":
            # Multilingual model even when forced: a caller asking for
            # "faster-whisper" wants Whisper, not an English-only distillation of it.
            engine = FasterWhisperEngine(
                model=MULTILINGUAL_MODEL_CUDA if hw.nvidia_cuda else MULTILINGUAL_MODEL_CPU,
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
        if force == "parakeet-mlx":
            from lattice_asr.engines.parakeet_mlx import ParakeetMlxEngine

            engine = ParakeetMlxEngine()
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
        from lattice_asr.engines.parakeet_mlx import ParakeetMlxEngine
        from lattice_asr.engines.whisper_cpp import WhisperCppEngine

        return {
            "en": ParakeetMlxEngine(),
            "multi": WhisperCppEngine(),
        }

    if hw.nvidia_cuda and hw.cuda_capability is not None and hw.cuda_capability >= (7, 0):
        from lattice_asr.engines.parakeet_tdt import ParakeetTdtEngine

        return {
            "en": ParakeetTdtEngine(),
            "multi": FasterWhisperEngine(
                model=MULTILINGUAL_MODEL_CUDA, device="cuda", compute_type="float16"
            ),
        }

    # Two engines, not one shared instance: distil-large-v3 is fast and English-only,
    # so it can serve "en" but must not be what non-English audio lands on.
    return {
        "en": FasterWhisperEngine(model="distil-large-v3", device="cpu", compute_type="int8"),
        "multi": FasterWhisperEngine(
            model=MULTILINGUAL_MODEL_CPU, device="cpu", compute_type="int8"
        ),
    }


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
        """Construct hardware-adaptive Transcriber.

        When `enable_diarization=True`, loads the diarizer adapter named by
        `config.diarization.adapter` (defaults to 'pyannote'; 'sortformer' also supported).
        """
        self._default_language = default_language
        self._config = config or LatticeAsrConfig()
        self._telemetry = telemetry_sink or NullTelemetrySink()
        self._enable_diarization = enable_diarization
        self._hardware = detect_hardware()
        self._engines = _build_engine_registry(
            self._hardware, force_engine or self._config.hardware_force
        )
        # Language detection is a capability OF AN ENGINE now, not a separate
        # model. Silero is gone (see lattice_asr.lid); whichever engine can
        # detect does the detecting.
        self._lid_enabled = self._config.lid.enabled
        self._detector = self._pick_detector() if self._config.lid.enabled else None
        self._diarizer = None
        if enable_diarization:
            if self._config.diarization.adapter == "pyannote":
                from lattice_asr.diarize.pyannote import PyAnnoteAdapter

                self._diarizer = PyAnnoteAdapter(
                    model=self._config.diarization.pyannote.model,
                    auth_token=os.environ.get(self._config.diarization.pyannote.auth_token_env),
                )
            elif self._config.diarization.adapter == "sortformer":
                from lattice_asr.diarize.sortformer import NvidiaSortformerAdapter

                self._diarizer = NvidiaSortformerAdapter(
                    model=self._config.diarization.sortformer.model,
                )
            else:
                raise ValueError(f"unknown diarization.adapter: {self._config.diarization.adapter}")

    def _stamp_language_provenance(
        self,
        result: TranscriptionResult,
        requested: str | None,
        detection: LidResult | None,
        warning: str | None,
    ) -> TranscriptionResult:
        """Record what the language is AND how we know, filling gaps last-resort.

        Order of authority: an explicit request, then the engine's own report,
        then an audio detector, then -- only if all of those came up empty -- the
        language of the transcribed text. Text LID is last because it describes
        what the engine WROTE rather than what was SPOKEN.
        """
        from dataclasses import replace as _replace

        language = result.language
        confidence = result.confidence
        warnings = [warning] if warning else []

        if requested is not None:
            source = "requested"
        elif detection is not None and detection.is_determined:
            # Detection is checked BEFORE the engine's own report on purpose. When a
            # detector determines the language we PASS it to the engine, and engines
            # echo it back -- so crediting "engine" here would launder an audio-LID
            # result as the engine's own finding and lose the confidence with it.
            language = detection.language
            confidence = detection.confidence
            source = "audio-lid"
        elif language and language != UNDETERMINED:
            source = "engine"
        else:
            source = "undetermined"

        # Last resort: the engine transcribed successfully but cannot say in what
        # language (Parakeet TDT). Better a labelled text-derived answer than a
        # transcript of obvious Spanish stamped "und".
        if source == "undetermined" and result.text:
            from lattice_asr.textlid import detect_language_from_text

            text_result = detect_language_from_text(result.text)
            if text_result is not None:
                language, confidence = text_result.language, text_result.confidence
                source = "transcript-text"
                warnings.append(
                    f"language {language!r} was inferred from the TRANSCRIPT TEXT "
                    f"(confidence {confidence:.4f}), not from the audio: no loaded "
                    f"engine can report a language"
                )

        return _replace(
            result,
            language=language or UNDETERMINED,
            confidence=confidence,
            language_source=source,
            language_warning=("; ".join(warnings) if warnings else result.language_warning),
        )

    def _pick_detector(self) -> TranscriptionEngine | None:
        """Choose which loaded engine performs language identification.

        Prefer the widest-coverage engine that actually overrides
        ``detect_language``. Returns None when no loaded engine can detect --
        a real state on hosts where only an auto-detecting engine is installed,
        and one the caller must handle rather than paper over.
        """
        candidates = [
            e
            for e in dict.fromkeys(self._engines.values())
            if type(e).detect_language is not TranscriptionEngine.detect_language
            and type(e).is_available()
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: len(e.capabilities.languages))

    def _select_engine(self, language: str | None) -> TranscriptionEngine:
        """Pick the engine for `language`, routing by CAPABILITY not by name.

        Replaces the old ``route = "en" if language == "en" else "multi"``. That
        line was the reason LID had to run up front at all, and it hard-coded a
        two-route world: an engine's real ``capabilities.languages`` was never
        consulted, so a multilingual engine registered under "en" could not be
        used for anything else, and an unknown language was sent to whichever
        engine happened to sit at "multi" -- on hosts where that engine's
        dependency is not installed, straight into ModuleNotFoundError.

        `language is None` means "unknown", and the SAFE answer for unknown is
        the broadest engine available, never the narrowest.
        """
        declared = list(dict.fromkeys(self._engines.values()))
        if not declared:
            raise EngineUnavailableError("engine registry is empty; nothing can transcribe")

        # PREFER available engines; do not hard-fail on none being available.
        # Every engine here is an OPTIONAL extra (pyproject: whisper/parakeet/
        # nvidia), so "no engine's package is installed" is a supported state of
        # this library -- and a caller may legitimately have injected or patched
        # an engine we cannot introspect. What matters is that we never pick an
        # uninstalled engine while an installed one can do the job: that is the
        # link failure mode, where the multilingual route pointed at
        # faster-whisper which was not installed, so any non-English audio would
        # have raised ModuleNotFoundError instead of transcribing.
        engines = [e for e in declared if type(e).is_available()]
        if not engines:
            logger.warning(
                "lattice-asr: NO registered engine has its dependencies installed "
                "(%s). Proceeding with the declared engine, which will fail on use "
                "unless it was injected or patched. Install an extra: "
                "`pip install 'lattice-asr[whisper]'` or `[nvidia]`.",
                ", ".join(
                    f"{e.capabilities.name} missing {','.join(type(e).missing_packages()) or 'n/a'}"
                    for e in declared
                ),
            )
            engines = declared
        if language is not None and language != UNDETERMINED:
            exact = [e for e in engines if language in e.capabilities.languages]
            if exact:
                # Narrowest sufficient engine wins: a specialist on its own
                # language is normally the faster one (that is the whole point
                # of the "en" route existing).
                return min(exact, key=lambda e: len(e.capabilities.languages))
            logger.warning(
                "lattice-asr: no loaded engine declares support for language %r "
                "(loaded: %s); falling back to the broadest engine",
                language,
                {e.capabilities.name: sorted(e.capabilities.languages)[:6] for e in engines},
            )
        return max(engines, key=lambda e: len(e.capabilities.languages))

    async def _resolve_language(
        self, audio_pcm: bytes, sample_rate: int, requested: str | None
    ) -> tuple[str | None, LidResult | None, str | None]:
        """Decide the language to transcribe in.

        Returns ``(language, detection, warning)`` where ``language is None``
        means "let the engine decide for itself" -- an outcome the old code
        could not express, and the absence of which caused this whole defect.

        **The fallback direction is inverted from the original on purpose.**
        Before: a broken/absent/low-confidence detector fell back to
        ``default_language`` ("en") and routed to the English-only engine, so
        the LEAST certain inputs were sent to the LEAST capable engine and
        silently mangled. Now uncertainty resolves to ``None``, which
        ``_select_engine`` routes to the broadest engine, and every such
        decision carries a warning the caller can see. Brief §5: a detector
        that fails closed is worth more than one that fails accurate.
        """
        if requested is not None:
            return requested, None, None

        if not self._lid_enabled:
            return (
                None,
                None,
                (
                    "language detection is disabled (config.lid.enabled=false) and no "
                    "language was requested; the engine auto-detected"
                ),
            )

        if self._detector is None:
            return (
                None,
                None,
                (
                    "no loaded engine can identify language; the engine auto-detected "
                    "without a confidence score"
                ),
            )

        try:
            detection = await self._detector.detect_language(audio_pcm, sample_rate)
        except Exception as exc:  # noqa: BLE001 - a broken detector must not fail the transcription
            # The 2026-05-27 Silero breakage raised here and was swallowed into
            # "English". It is still not fatal, but it is no longer silent.
            logger.warning(
                "lattice-asr: language detection FAILED (%s: %s); "
                "transcribing with engine auto-detect instead",
                type(exc).__name__,
                exc,
            )
            return None, None, f"language detection failed: {type(exc).__name__}: {exc}"

        if detection is None or not detection.is_determined:
            return None, detection, "language detector returned no determination"

        threshold = self._config.lid.confidence_threshold
        if detection.confidence < threshold:
            warning = (
                f"low-confidence language detection: {detection.language!r} at "
                f"{detection.confidence:.4f} < threshold {threshold}; "
                f"routed to engine auto-detect rather than assuming "
                f"{self._default_language!r}"
            )
            logger.warning("lattice-asr: %s", warning)
            return None, detection, warning

        return detection.language, detection, None

    @property
    def hardware(self) -> HardwareProfile:
        """Detected `HardwareProfile` (OS, arch, accelerator, RAM/cores) for engine selection."""
        return self._hardware

    @property
    def loaded_engines(self) -> dict[str, TranscriptionEngine]:
        """Map language route ("en"/"multi") -> engine; values may share an instance on CPU."""
        return dict(self._engines)

    async def warmup(self) -> None:
        """Warm every loaded engine.

        No separate LID warmup: the detector IS one of these engines now.
        """
        await asyncio.gather(*(e.warmup() for e in set(self._engines.values())))

    async def transcribe(
        self,
        audio_pcm: bytes,
        sample_rate: int = 16000,
        *,
        language: str | None = None,
        diarize: bool = False,
        tenant_id: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe PCM audio.

        Routes via LID-or-explicit (spec §8.2) and records telemetry. Raises ValueError
        if `diarize=True` was passed without `enable_diarization=True` at construction.
        """
        if diarize and not self._enable_diarization:
            raise ValueError("diarize=True requires enable_diarization=True at __init__")

        requested = language
        language, detection, warning = await self._resolve_language(
            audio_pcm, sample_rate, requested
        )
        engine = self._select_engine(language)
        result = await engine.transcribe(audio_pcm, sample_rate, language)

        result = self._stamp_language_provenance(result, requested, detection, warning)

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
                # Confidence in the LANGUAGE decision specifically -- the
                # detector's number when one ran, the text-LID number when that
                # was the fallback, and None when nobody measured the language
                # (a pinned request, or an engine that reports none).
                language_confidence=(
                    detection.confidence
                    if detection is not None
                    else (
                        result.confidence if result.language_source == "transcript-text" else None
                    )
                ),
                language_warning=result.language_warning,
                language_source=result.language_source,
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
        """Stream partial transcriptions.

        No per-chunk LID (deferred); one telemetry record per partial. Same diarize gate
        as transcribe(), but the ValueError is deferred to the first iteration because
        this is an async generator.
        """
        if diarize and not self._enable_diarization:
            raise ValueError("diarize=True requires enable_diarization=True at __init__")
        # No per-chunk LID (a 1s window is too little audio to identify a
        # language from). But an absent hint stays None rather than becoming
        # `default_language`: that substitution is exactly the silent-English
        # defect, and here it would also pin a whole stream to the wrong engine.
        lang = language
        engine = self._select_engine(lang)
        if lang is None:
            logger.info(
                "lattice-asr: streaming with no language hint; routed to %s for auto-detect",
                engine.capabilities.name,
            )
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
